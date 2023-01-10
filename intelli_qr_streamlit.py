import streamlit as st
import streamlit_ext as ste
from PIL import Image
import utils_evil
import argparse
from io import BytesIO

st.set_page_config(page_title="Intelli-QR")

# Page title
st.title("Intelli-QR")
st.write("## Styled QR Code Generator")
st.write("(or just QR Code Generator)")
st.write("Note: please be patient when there is \"RUNNING...\" on the upper right of the page.")
st.write("It takes a while to process the image.")


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

qr_code = None
st.subheader("Enter the url you want to generate:")
url = st.text_input("URL", "")
if url != "":
    qr_code, qr_version = utils_evil.generate_qr_code(url, module_size=16)
    st.image(qr_code, caption="QR Code", width=256)

tab_1, tab_2 = st.tabs(["Styled QR", "NOT-Styled QR"])

with tab_1: # Styled QR Code
    upload_column_1, upload_column_2 = st.columns(2)
    with upload_column_1:
        st.subheader("Content image:")
        upload_content_img = st.file_uploader("Upload content image here (png or jpg)", type=['png', 'jpg'])
    with upload_column_2:
        st.subheader("Style image:")
        upload_style_img = st.file_uploader("Upload style image here (png or jpg)", type=['png', 'jpg'])


    content_column, style_column = st.columns(2)
    if upload_content_img is not None:
        input_content_image = Image.open(upload_content_img).convert("RGB").resize([512, 512], Image.Resampling.LANCZOS)

        #Want to replace input image so as not to take up space with each new one 
        with content_column:
            st.image(input_content_image, caption='Content Image', use_column_width=True)
        
    if upload_style_img is not None:
        input_style_img = Image.open(upload_style_img).convert("RGB").resize([512, 512], Image.Resampling.LANCZOS)
        
        with style_column:
            st.image(input_style_img, caption='Style Image', use_column_width=True)
            
    # Pick NST method
    st.subheader('Select a NST method: ')
    nst_method = st.selectbox('select NST method', ('Stytr2', 'AdaIN', 'Mast', 'SANet'))

    if upload_content_img is not None and upload_style_img is not None and qr_code is not None:
        
        OUTPUT_PATH = './outputs-streamlit/'
        
        def get_args():
            parser = argparse.ArgumentParser()
            parser.add_argument('--nst', type=str, default='stytr2',
                                help='Neural style transfer method.')

            parser.add_argument('--output_dir', type=str, default='output',
                                help='Directory to save the output image(s)')
            parser.add_argument('--output_name', type=str, default="output")

            parser.add_argument('--decoder_path', type=str,
                                default='./nst_StyTr2/models/decoder.pth')
            parser.add_argument('--transform_path', type=str,
                                default='./nst_StyTr2/models/transformer_iter_160000.pth')
            parser.add_argument('--vgg_path', type=str,
                                default='./nst_StyTr2/models/vgg_normalised.pth')

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
        args = parser.parse_args("")

        
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

        output_qr_colored = None
        
        if st.button('Generate Styled QR!'):
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
            st.image(output_qr_colored, width=400)
            st.write("You can right click to save the code now!")

        if output_qr_colored is not None:
            ste.download_button(label="Download QR",
                        data=convert_image(output_qr_colored),
                        file_name="intelli_qr.jpg",
                        mime="image/jpg")
        
with tab_2: # NOT-styled QR Code
    st.subheader("Background image: ")
    upload_background_image = st.file_uploader("Upload background image here (png or jpg)", type=['png', 'jpg'])
    if upload_background_image is not None:
        background_image = Image.open(upload_background_image).convert("RGB")
        st.image(background_image, caption='Content Image', width=400)
        
    output_qr_colored = None
        
    if (qr_code is not None) and (upload_background_image is not None):
        code_size = (4 * qr_version + 17) * 16
        
        if st.button('Generate QR!'):
            background_image = background_image.resize([code_size, code_size], Image.Resampling.LANCZOS)
            qr_code_colored = utils_evil.colorize_code(background_image, qr_code)
            arr = utils_evil.embbed_qr_rgb(background_image,
                                        qr_code, 
                                        module_size=16,
                                        qr_version=qr_version)
            output_qr_colored = Image.fromarray(arr).convert("RGB")
            output_qr_colored = utils_evil.add_pattern(output_qr_colored,
                                                       qr_code_colored,
                                                       qr_version=qr_version,
                                                       module_size=16)
            output_qr_colored = utils_evil.add_border(output_qr_colored, 8)
            st.image(output_qr_colored, width=400)
            st.write("You can right click to save the code now!")

            if output_qr_colored is not None:
                ste.download_button(label="Download QR",
                            data=convert_image(output_qr_colored),
                            file_name="intelli_qr.jpg",
                            mime="image/jpg")
            
