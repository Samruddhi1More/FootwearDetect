from footwear_detect import DetectClasses
import streamlit as st

models = DetectClasses()

# def defect_type(image_file):
#     result = model.predict(image_file)
#     return result.title()

def step_wise_pred(image_file):
    pred = models.predict(image_file)
    return pred

def main():
    st.title('Footwear Type Detector')
    uploaded_file = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        # st.image(uploaded_file, caption='Uploaded Image', width=300)
        st.markdown("### Preview")
        col1, col2, col3 = st.columns(3)

        with col1:  # to render the image in the middle
            st.write(' ')

        with col2:
            st.image(uploaded_file, width=400)

        with col3:
            st.write(' ')
            
        st.markdown("### Result")

        pred = step_wise_pred(uploaded_file)
        st.success(f"Detected: {pred}")
        # st.success(f"Defect Type: {defect}")

        #Display prediction
        # st.write(f'Defect Type: {defect_type(uploaded_file)}')

if __name__ == "__main__":
    main()