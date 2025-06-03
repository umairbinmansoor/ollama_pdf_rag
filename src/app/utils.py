import streamlit as st
import logging
import base64
from groq import Groq
import os
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
import re
#############################
# Headers and Prompts
#############################
headers = {
    "authorization":st.secrets["GROQ_API_KEY"],
    "content-type":"application/json"
}

#################################
# Function to encode the image
#################################
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
########################################################################
# Function to extract JSON from images using Groq API
########################################################################
def JSON_extractor(images_mapping, output_dir_img, groq_api_key):
    client = Groq(api_key=groq_api_key)

    # Pre-encode all images and collect captions
    encoded_inputs = []
    for label, info in images_mapping.items():
        image_path = os.path.join(output_dir_img, info["filename"])
        caption_text = info['caption']
        try:
            with open(image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
                encoded_inputs.append((label, encoded_img, caption_text))
        except Exception as e:
            logging.error(f"Failed to encode image '{label}' from '{image_path}': {e}")

    def inference(label, encoded_img, caption_text):
        # Include caption in prompt if available
        caption_note = f'The caption reads: "{caption_text.strip()}"\n\n' if caption_text else ""

        prompt_text = (
            f"The image is labeled '{label}'.\n"
            f"{caption_note}"
            "If the image contains a figure, table, chart, or diagram, extract the key information in a structured JSON format. "
            "Your output must be a valid JSON object with the following fields:\n"
            "- image_label\n"
            "- type (e.g., figure, table, chart, diagram)\n"
            "- title\n"
            "- description\n"
            "- data_summary\n"
            "- key_components\n"
            "- observations\n\n"
            "Be concise and accurate. If a field is not applicable, use null or an empty string."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}},
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=0.5,
                max_completion_tokens=4096,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
            )
            return (label, response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error processing image '{label}': {e}")
            return (label, None)

    # Parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        results_with_labels = list(executor.map(lambda args: inference(*args), encoded_inputs))

    # Filter only successful results
    successful_results = [content for label, content in results_with_labels if content is not None]

    return successful_results

################################################################################
# Function to extract images from PDF and map to labels
################################################################################
# Step 1: Extract images from PDF and map to labels
def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    images_mapping = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Get image bounding box
            try:
                image_rect = page.get_image_bbox(img)
            except Exception as e:
                print(f"Warning: Could not get bounding box for image {xref} on page {page_num + 1}: {e}")
                image_rect = None

            # Extract caption (assuming it's below the image or nearby)
            caption_text = ""
            if image_rect:
                # Define a region below the image for caption
                # Try below first
                caption_region = fitz.Rect(image_rect.x0 - 20,
                                           image_rect.y1 + 5,
                                           image_rect.x1 + 10,
                                           image_rect.y1 + 80)
                caption_text = page.get_text("text", clip=caption_region, sort=True).strip()

                # If not found, try above
                if not caption_text.strip():
                    caption_region_above = fitz.Rect(image_rect.x0 - 10, image_rect.y0 - 80, image_rect.x1 + 10, image_rect.y0 - 5)
                    caption_text = page.get_text("text", clip=caption_region_above, sort=True).strip()


            # Fallback: Extract text from the entire page if no bounding box or caption found
            if not caption_text:
                caption_text = page.get_text("text")

            # Search for caption pattern
            # caption_match = re.search(r"(Fig|Figure|Table) (\d+):[^\n]+", caption_text, re.IGNORECASE)
            # caption_match = re.search(
            #         r"(Figure|Fig\.?|Table|Tab\.?)\s*(\d+)\s*[:.\-–\s]*\s*(.+)",
            #         caption_text,
            #         re.IGNORECASE | re.DOTALL
            #     )
            caption_match = re.search(
                      r"(Figure|Fig\.?|Table|Tab\.?)\s*(\d+(?:\.\d+)+|\d+)\s*[:.\-–\s]*\s*(.+)",
                      caption_text,
                      re.IGNORECASE | re.DOTALL
                  )


            # print(caption_text)
            if caption_match:
                prefix = caption_match.group(1).replace(".", "")  # e.g., 'Fig.' → 'Fig'
                number = caption_match.group(2)
                label = f"{prefix}{number}"                      # e.g., 'Fig7'
                full_caption = caption_match.group(3).strip()
            else:
                label = f"page{page_num + 1}_img{img_index + 1}"
                full_caption = caption_text.strip()  # Use the full caption text if no match found

            # Save image
            image_filename = f"{label}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Store mapping
            images_mapping[label] = {
                "filename": image_filename,
                "caption": full_caption
            }

    doc.close()
    return images_mapping
#####################################################################################################