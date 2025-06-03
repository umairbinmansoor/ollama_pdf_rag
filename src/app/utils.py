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
    # client = Groq(api_key=groq_api_key)
    client = Groq(api_key=headers["authorization"]) # Extract API key from headers

    # Pre-encode all images
    encoded_images = {
        label: encode_image(os.path.join(output_dir_img, img_name))
        for label, img_name in images_mapping.items()
    }

    def inference(label, encoded_img):
        # Refine the prompt to guide the model on what kind of structured information to extract
        # Added instructions to specifically look for figures, tables, charts and extract key details as JSON
        prompt_text = ("Analyze the image. Extract any unstructured information in the JSON format."
        # prompt_text = (
        #     "Analyze the image. If it contains a figure, table, or chart, "
        #     "extract the key information, title, axes labels, data points (if possible "
        #     "and relevant, summarize if too many), and any significant observations or trends. "
        #     "If it's a complex diagram or illustration, describe its main components and purpose. "
        #     "Return the extracted information as a JSON object. "
        #     "Ensure the JSON is valid and well-formed. "
        #     "Use keys like 'image_label', 'type' (e.g., 'figure', 'table', 'chart', 'diagram'), "
        #     "'title', 'description', 'data_summary', 'key_components', etc., relevant to the image content."
             f"The label for this image is '{label}'." # Include label in prompt for context
         )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_img}",
                        },
                    },
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=0.5, # Lower temperature for more focused output
                max_completion_tokens=4096,
                top_p=1,
                stream=False,
                # Keep response_format as json_object, but rely more on prompt guidance
                response_format={"type": "json_object"},
                stop=None,
            )
            # Return a tuple of (label, content) to associate JSON with its image
            return (label, response.choices[0].message.content)
        except Exception as e: # Catch any exception during the API call
            logging.error(f"Error processing image '{label}': {e}")
            # Return None or a specific error indicator for this image
            return (label, None) # Return label and None for failed items

    # Parallel processing (CPU-bound)
    # Use a list comprehension to filter out None results later if needed
    with ThreadPoolExecutor(max_workers=5) as executor:
        # map returns an iterator, convert to list to trigger execution
        results_with_labels = list(executor.map(lambda kv: inference(*kv), encoded_images.items()))

    # Filter out failed results and return only the JSON content
    # Or, keep labels if needed for downstream processing
    successful_results = [content for label, content in results_with_labels if content is not None]

    return successful_results # Return only the successful JSON strings

################################################################################
# Function to extract images from PDF and map to labels
################################################################################
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

            # Try to get bounding box
            try:
                image_rect = page.get_image_bbox(img)
            except Exception as e:
                logging.error(f"Could not get bounding box for image {xref} on page {page_num + 1}: {e}")
                image_rect = None

            caption_text = ""
            if image_rect:
                # Extend region below image for multi-line caption (e.g. up to 5 lines)
                caption_region = fitz.Rect(
                    image_rect.x0,
                    image_rect.y1 + 10,
                    image_rect.x1,
                    image_rect.y1 + 100  # Adjust as needed
                )
                caption_text = page.get_text("text", clip=caption_region, sort=True).strip()

            if not caption_text:
                caption_text = page.get_text("text")

            # Updated regex: supports Fig., Tab., multi-line after match
            caption_match = re.search(
                                r"(Figure|Fig\.|Table|Tab\.)\s*(\d+)\s*[:.\s-]*\s*(.+)",
                                caption_text, re.IGNORECASE | re.DOTALL
                            )
            if caption_match:
                prefix = caption_match.group(1).replace(".", "")  # e.g., 'Fig.' → 'Fig'
                number = caption_match.group(2)
                label = f"{prefix}{number}"                      # e.g., 'Fig7'
                full_caption = caption_match.group(0).strip()
            else:
                label = f"page{page_num + 1}_img{img_index + 1}"
                full_caption = ""

            # Save image
            # image_filename = f"{label.replace(' ', '_')}.{image_ext}"
            image_filename = f"{label}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            images_mapping[label] = {
                "image_filename": image_filename,
                "caption": full_caption
            }

    doc.close()
    return images_mapping
#####################################################################################################