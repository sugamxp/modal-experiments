def main():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import requests
    
    st.title("Modal Project: Stable Diffusion XL")

    sdxl_form = st.form(key='sdxl_form')
    prompt = sdxl_form.text_input('Enter your prompt', 'An astronaut riding a purple horse')
    submit = sdxl_form.form_submit_button('Submit')
    SDXL_URL = "https://sugamxp--stable-diffusion-xl-model-web-inference.modal.run/"
    
    if submit:
      with st.spinner("Wait for it..."):
        response = requests.get(SDXL_URL, params={"prompt": prompt})
        image_bytes = response.content
        st.image(image_bytes, caption=prompt)

if __name__ == "__main__":
    main()