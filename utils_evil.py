import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageChops, ImageOps
from torchvision.utils import make_grid
import qrcode
import math
import cv2
import torch

def add_border(input: Image, border: int):
    return ImageOps.expand(input, border=border, fill='white')


def add_pattern(target_PIL, code_PIL, qr_version=5, module_size=16):
    def get_alignment_positions(version):
        positions = []
        if version > 1:
            n_patterns = version // 7 + 2
            first_pos = 6
            positions.append(first_pos)
            matrix_width = 17 + 4 * version
            last_pos = matrix_width - 1 - first_pos
            second_last_pos = (
                (first_pos + last_pos * (n_patterns - 2)  # Interpolate end points to get point
                + (n_patterns - 1) // 2)                  # Round to nearest int by adding half
                                                        # of divisor before division
                // (n_patterns - 1)                       # Floor-divide by number of intervals
                                                        # to complete interpolation
                ) & -2                                    # Round down to even integer
            pos_step = last_pos - second_last_pos
            second_pos = last_pos - (n_patterns - 2) * pos_step
            positions.extend(range(second_pos, last_pos + 1, pos_step))
        return positions
    target_img = np.asarray(target_PIL)
    code_img = np.asarray(code_PIL)
    output = target_img
    output = np.require(output, dtype='uint8', requirements=['O', 'W'])
    m_size = module_size
    m_number = 4 * qr_version + 17

    # Alignment
    alignment_loactions = get_alignment_positions(qr_version)
    # alignment_loactions.pop(0) # covered by the upper left finder pattern
    for i in range(len(alignment_loactions)):
        for j in range(len(alignment_loactions)):
            alignment = alignment_loactions[i]
            alignment_ = alignment_loactions[j]
            if i == 0:
                if j == 0 or j == len(alignment_loactions) - 1:
                    continue
            elif i == len(alignment_loactions) - 1 and j == 0:
                continue
            
            align_begin = alignment - 2
            align_begin_ = alignment_ - 2
            output[align_begin * m_size: (align_begin + 5) * m_size, align_begin_ * m_size:(align_begin_ + 5) * m_size, :] = \
                code_img[align_begin * m_size: (align_begin + 5) * m_size, align_begin_ * m_size:(align_begin_ + 5) * m_size, :]    
    
    # Finder
    # upper left
    output[0:(8 * m_size) , 0:(8 * m_size), :] = \
        code_img[0:(8 * m_size) , 0:(8 * m_size), :]

    # upper right
    output[((m_number - 8) * m_size):(m_number * m_size), 0:(8 * m_size) , :] = \
        code_img[((m_number - 8) * m_size):(m_number * m_size), 0:(8 * m_size) , :]

    # lower left
    output[0: (8 * m_size) , ((m_number - 8) * m_size):(m_number * m_size), :] = \
        code_img[0: (8 * m_size) , ((m_number - 8) * m_size):(m_number * m_size), :]
        
    output = Image.fromarray(output.astype('uint8'))
    return output


def tensor_to_PIL(tensor):
    grid = make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def colorize_code(background_PIL: Image, code_PIL: Image, selected_colors=5):
    def k_means(pixels, n):
        cluster = KMeans(n_clusters=n, n_init=5)
        cluster.fit(pixels)
        centers = cluster.cluster_centers_

        clustered = cluster.predict(pixels)
        colors, counts = np.unique(clustered, axis=0, return_counts=1)
        return centers, colors, counts
    
    background_PIL = background_PIL.convert('RGB')
    code_PIL = code_PIL.convert('RGB')
    background_pixels = np.array(background_PIL).reshape((-1, 3))
    
    centers, colors, counts = k_means(background_pixels, selected_colors)
    
    sorted_pairs = sorted(zip(counts, colors), reverse=True)
    tuples = zip(*sorted_pairs)
    counts, colors = [list(tuple) for tuple in tuples]
    
    # contranst should > threshold
    contrast_thr = 320
    idx = 0
    qr_result = np.array(code_PIL)
    contrast_alert = True

    for idx in range(len(colors)):
        dominant_colors_index = colors[idx]
        if 255 * 3 - np.sum(centers[dominant_colors_index]) >= contrast_thr:
            contrast_alert = False
            break
    
    if contrast_alert:
        dominant_colors_index = colors[0]
        
    black_module_index = np.where((qr_result < [128, 128, 128]).all(
        axis=2))
    qr_result[black_module_index] = centers[dominant_colors_index]
    
    if contrast_alert:
        qr_result = cv2.cvtColor(qr_result, cv2.COLOR_RGB2HLS_FULL)
        qr_result[black_module_index[0],black_module_index[1], 1] = 120
        qr_result = cv2.cvtColor(qr_result, cv2.COLOR_HLS2RGB_FULL)
    
    qr_result = Image.fromarray(qr_result)
    return qr_result


def embbed_qr_rgb(img, code, module_size=16, qr_version=5):
    m_size = module_size
    m_num = 4 * qr_version + 17
    if m_size % 2 == 0:
        center_radius = (math.floor(module_size / 3) + 1) // 2
        center_index = m_size // 2
    else:
        center_radius = math.floor(module_size / 3) // 2
        center_index = m_size // 2
        
    
    code_arr = np.asarray(code)
    cv_img_arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HLS_FULL)
    for i in range(m_num):
        for j in range(m_num):
            module = code_arr[i * m_size:(i + 1) * m_size, j * m_size:(j + 1) * m_size]
            start = center_index - center_radius
            end = center_index + center_radius
            module_color = np.mean(module[start:end, start:end])
            center_luminance = cv_img_arr[i * m_size + start:i * m_size + end, j * m_size + start:j * m_size + end, 1] # module center
            if module_color < 127:
                if np.mean(center_luminance) > 60:
                    cv_img_arr[i * m_size + start:i * m_size + end, j * m_size + start:j * m_size + end, 1] = 60
            else:
                if np.mean(center_luminance) < 190:
                    cv_img_arr[i * m_size + start:i * m_size + end, j * m_size + start:j * m_size + end, 1] = 190
    return cv2.cvtColor(cv_img_arr, cv2.COLOR_HLS2RGB_FULL)


def generate_qr_code(qr_data, qr_version=5, module_size=16):
	qr = qrcode.QRCode(
		version=qr_version,
		error_correction=qrcode.constants.ERROR_CORRECT_L,
		box_size=module_size,
		border=0,
	)

	qr.add_data(qr_data)
	qr.make(fit=True)
	code_img = qr.make_image(fill='black', back_color='white') \
		.convert('RGB')
	return code_img, qr.version


def trim(code_PIL):
	bg = Image.new(code_PIL.mode, code_PIL.size, code_PIL.getpixel((0, 0)))
	diff = ImageChops.difference(code_PIL, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return code_PIL.crop(bbox)
