import streamlit as st
from PIL import Image
import utils_evil
import argparse


# Page title
st.title("Pytorch Neural Style Transfer On Streamlit")


# Pick NST method
nst_method = st.selectbox('Pick NST method: ', ('Stytr2', 'AdaIN', 'Mast', 'SANet'))
st.write('You selected:', nst_method)


QR_CODE_PATH = "./images/qr-images/userinput.png"
st.header("Enter the url you want to generate:")
qr_code = None
url = st.text_input("URL", "")
if url != "":
    qr_code, qr_version = utils_evil.generate_qr_code(url, module_size=16)
    # qr_code.save(QR_CODE_PATH)
    st.image(qr_code, caption="QR Code", width=256)

upload_column_1, upload_column_2 = st.columns(2)
with upload_column_1:
    st.header("Choose your content image:")
    upload_content_img = st.file_uploader("Upload content image here (png or jpg)", type=['png', 'jpg'])
with upload_column_2:
    st.header("Choose your style image:")
    upload_style_img = st.file_uploader("Upload style image here (png or jpg)", type=['png', 'jpg'])


content_column, style_column = st.columns(2)
if upload_content_img is not None:
    input_content_image = Image.open(upload_content_img).convert("RGB").resize([512, 512], Image.Resampling.LANCZOS)

    #Want to replace input image so as not to take up space with each new one 
    CONTENT_PATH = "./temp/input_content.jpg"
    with content_column:
        st.header("Content Image")
        st.image(input_content_image, caption='Content Image', use_column_width=True)
    # input_content_image.save(CONTENT_PATH)
    
if upload_style_img is not None:
    input_style_img = Image.open(upload_style_img).convert("RGB").resize([512, 512], Image.Resampling.LANCZOS)
    
    STYLE_PATH = "./temp/input_style.jpg"
    with style_column:
        st.header("Style Image")
        st.image(input_style_img, caption='Style Image', use_column_width=True)
    # input_style_img.save(STYLE_PATH)
        
    

if upload_content_img is not None and upload_style_img is not None and qr_code is not None:
    
    OUTPUT_PATH = './outputs-streamlit/'
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--nst', type=str, default='stytr2',
                            help='Neural style transfer method.')
        parser.add_argument('--content', type=str,
                            help='File path to the content image', required=True)
        parser.add_argument('--style', type=str,
                            help='File path to the style image, or multiple style \
                            images separated by commas if you want to do style \
                            interpolation or spatial control', required=True)

        parser.add_argument('--output_dir', type=str, default='output',
                            help='Directory to save the output image(s)')
        parser.add_argument('--output_name', type=str, default="output")

        parser.add_argument('--decoder_path', type=str,
                            default='./nst_StyTr2/models/decoder.pth')
        parser.add_argument('--transform_path', type=str,
                            default='./nst_StyTr2/models/transformer_iter_160000.pth')
        parser.add_argument('--vgg_path', type=str,
                            default='./nst_StyTr2/models/vgg_normalised.pth')

        # parser.add_argument('--style_interpolation_weights', type=str, default="")
        parser.add_argument('--content_size', type=int, default=512)
        parser.add_argument('--style_size', type=int, default=512)
        parser.add_argument('--crop', action='store_true',
                            help='do center crop to create squared image')
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--save_ext', default='.jpg',
                            help='The extension name of the output image')


        # 2nd stage
        parser.add_argument('--code_data', type=str,
                            default="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        parser.add_argument('--code_module_size', type=int,
                            default=16)
        parser.add_argument('--code_image_name', type=str,
                            default="code_output.jpg")
        parser.add_argument('--code_output_dir', type=str,
                            default="./output/")
        return parser

    parser = get_args()
    args = parser.parse_args("--content {} \
        --style {}".format(CONTENT_PATH, STYLE_PATH).split())

    
    if str.lower(nst_method) == "stytr2": #default
        from nst_StyTr2.test import style_transfer_with_input
    elif str.lower(nst_method) == "adain":
        from nst_adain.test import style_transfer_with_input
        args.decoder_path = "./nst_adain/models/decoder.pth"
    elif str.lower(nst_method) == "mast":
        from nst_mast.test import style_transfer_with_input
        args.decoder_path = "./nst_mast/models/decoder_iter_160000.pth"
        args.transform_path = "./nst_mast/models/ma_module_iter_160000.pth"
    elif str.lower(nst_method) == "sanet":
        from nst_SANet.test import style_transfer_with_input
        args.decoder_path = "./nst_SANet/models/decoder.pth"
        args.transform_path = "./nst_SANet/models/transformer_iter_500000.pth"


    code_size = (4 * qr_version + 17) * args.code_module_size

    
    if st.button('Create Style Transfer QR Code'):
        background_image = utils_evil.tensor_to_PIL(style_transfer_with_input(args, input_content_image, input_style_img)).convert("RGB").resize([code_size, code_size], Image.Resampling.LANCZOS)
        qr_code_colored = utils_evil.colorize_code(background_image, qr_code)
        arr = utils_evil.embbed_qr_rgb(background_image,
                                       qr_code, 
                                       module_size=args.code_module_size,
                                       qr_version=qr_version)
        output_qr_colored = Image.fromarray(arr).convert("RGB")
        output_qr_colored = utils_evil.add_pattern(output_qr_colored,
                                                   qr_code_colored,
                                                   qr_version=qr_version,
                                                   module_size=args.code_module_size)
        output_qr_colored = utils_evil.add_border(output_qr_colored, 8)
        st.image(output_qr_colored, use_column_width=True)
        st.write("You can right click to save the code now!")
        
    # else:
    #     st.write('click button to run model')

