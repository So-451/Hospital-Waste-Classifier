import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import os
import pandas as pd





# Load model
model = YOLO("Hospital_Waste_Model.pt")

# Bin color mapping
bin_mapping = {
    "Yellow bin": "Infectious Waste",
    "Black bin": "General Waste",
    "Red bin": "Plastic Waste",
    "Blue bin": "Glass Waste"
}
# Disposal instructions for each bin
disposal_instructions = {
    "Yellow bin": "Dispose of infectious waste like blood-soaked items using autoclaving or incineration.",
    "Black bin": "General waste should be landfilled after basic disinfection if needed.",
    "Red bin": "Plastic waste must be disinfected and shredded before final disposal.",
    "Blue bin": "Glass waste should be disinfected and sent for recycling or deep burial."
}


st.set_page_config(page_title="Hospital Waste Classifier", layout="centered")


st.title("🏥 Hospital Waste Classifier")
st.markdown("Upload hospital waste images and classify them to the appropriate bin.")

# Store results
all_detections = []

# Upload images
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        # Load and show image
        image = Image.open(temp_file_path).convert("RGB")
        st.image(image, caption=f"📸 Uploaded: {uploaded_file.name}", use_container_width=True)

        with st.spinner("Detecting waste..."):
        

        # Run prediction
            results = model.predict(temp_file_path, conf=0.32)[0]

        if len(results.boxes) > 0:
            img_np = np.array(image)

            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results.names[cls_id]
                waste_type = bin_mapping.get(label, "Unknown")
                
                disposal_tip = disposal_instructions.get(label, "No instructions available.")



                # Draw box
                cv2.rectangle(img_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # After drawing each box
                cv2.rectangle(img_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)

                cv2.putText(img_np, f"{label} ({conf:.2f})", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                

                st.success(f"🗑️ Detected: **{label}**")
                st.info(f"Category: **{waste_type}**")
                st.warning(f"Confidence: **{conf:.2%}**")
                st.info(f"🧾 Disposal Tip: {disposal_tip}")
                all_detections.append({
                    "Image": uploaded_file.name,
                    "Label": label,
                    "Category": waste_type,
                    "Confidence": round(conf, 4)
                })

            st.image(img_np, caption="🧠 Detected Image with Bounding Boxes", use_container_width=True)
        else:
            st.error("No waste object detected or classification failed.")

        temp_file.close()
        os.unlink(temp_file_path)

# 📈 Summary Dashboard
if all_detections:
    df = pd.DataFrame(all_detections)

    st.subheader("📊 Waste Category Summary")
    chart_data = df["Category"].value_counts()
    st.bar_chart(chart_data)

    # 📥 Download predictions
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv", key="download-csv")

# Show history log of detections
if all_detections:
    with st.expander("📚 View Detection History"):
        st.dataframe(pd.DataFrame(all_detections))

