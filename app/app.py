#!/usr/bin/env python
"""Codegen Streamlit App
"""
import dotenv
from retry import retry

import streamlit as st
from retry import retry

import logging

logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

import json
import requests

def parse_sse_and_extract_json(sse_content):
    events = sse_content.strip().split("\n\n")
    for event in events:
        lines = [line for line in event.split("\n") if line]  # Remove empty lines
        for i, line in enumerate(lines):
            if line.startswith("event: data"):
                # Assuming the very next line after "event: data" contains the JSON data
                if i + 1 < len(lines) and lines[i+1].startswith("data:"):
                    try:
                        json_data = json.loads(lines[i+1][len("data:"):].strip())
                        return json_data
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        return None
    return None

@retry(tries=3, delay=0, backoff=1)
def get_code(message):
    response = requests.post(
        "http://localhost:8000/codelangchain/stream",
        json={
            "input": message
        }
    )
    return parse_sse_and_extract_json(response.text)



import boto3
import json
import base64

bedrock = boto3.client('bedrock-runtime')


def describe_image(image):
    """
    Describe an image using the Bedrock API
    """
    # Convert to base64 encoding
    image_base64 = base64.b64encode(image).decode('utf-8')

    # # Body for Claude v3 Sonnet
    body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10000,
    "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "image",
                "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64
                }
            },
            {
                "type": "text",
                "text": """Your task is to analyze the diagram to capture all of the AWS resources, configurations and components it depicts,
    along with the relationships between them. The end goal is to gather the necessary information to
    create tasks for Cloud Engineer to create the infrastructure-as-code."""
            }
            ]
        }
        ]
    })
    # Parameters for Claude V3 sonnet
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    accept = 'application/json'
    content_type = 'application/json'

    # Invoke Bedrock API
    response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)

    # Parse the response body
    response_body = json.loads(response.get('body').read())
    print(response_body)

    # Extract text
    text = response_body['content'][0]['text']
    return text


# Set the page title and icon
st.set_page_config(
    page_title="AI CodeGen",
    # page_icon=":robot:",
    page_icon="https://neurons-lab-public-bucket.s3.eu-central-1.amazonaws.com/Logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)





st.markdown('''
<style>
.page-subtitle_large {
   font-family: Helvetica;
   font-size: 40px;
   line-height: 41px;
   font-weight: bold;
   color: #121212;
   margin-bottom: 56px;
}

.highlighted {
   background: linear-gradient(90deg, rgba(163,216,255,1) 0%, rgba(182,136,245,1) 100%, rgba(2,0,36,1) 1000%);
   -webkit-background-clip: text;
   background-clip: text;
   color: transparent;
}
</style>


<div class="page-subtitle_large">Junior <span class="highlighted">AI Cloud Engineer</span> to create AWS Infrastructure as Code</div>
''', unsafe_allow_html=True)

# st.markdown("## Enter Website URL")

st.markdown('''
<style>
.page-subtitle_small {
   font-family: Helvetica;
   font-size: 24px;
   line-height: 41px;
   font-weight: bold;
   color: #121212;
   margin-bottom: 20px;
}
</style>


<div class="page-subtitle_small">Enter Website URL</div>
''', unsafe_allow_html=True)


with st.form("my_form"):

    input_prompt = st.text_area(
        label="Enter Prompt to Generate AWS CDK Infrastructure as Code",
        max_chars=1000,
        height=400,
        value="""Install the latest version of nginx, enable it and start the service on the EC2 instance
Install the instance in the existing VPC vpc-0fed7b21fa59b0985, in us-east-1 in 433559402488 aws account for the instance.
Secure the instance with EBS encryption."""
        )

    uploaded_file = st.file_uploader("Choose a AWS Architecture Diagram file", type=["jpg", "png"])

    submitted = st.form_submit_button("✨ Magic Button")


    if submitted:
        with st.spinner("Generating AWS CDK Infrastructure as Code..."):
            
            if uploaded_file:
                image = uploaded_file.read()
                st.image(
                    image,
                    caption='Uploaded AWS Architecture Diagram.', 
                    use_column_width=True,
                    output_format="auto"

                )
                image_description = describe_image(image)
            else:
                image_description=""
            result = get_code(
                f'{input_prompt}\n\n{image_description}',
            )
            st.text(result["prefix"])
            st.code(result["code"], language="python")

        