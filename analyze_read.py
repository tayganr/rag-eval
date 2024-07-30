# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# Source: https://github.com/Azure-Samples/document-intelligence-code-samples/blob/main/Python(v4.0)/Read_model/sample_analyze_read.py

"""
FILE: sample_analyze_read.py

DESCRIPTION:
    This sample demonstrates how to extract document information using "prebuilt-read"
    to analyze a given file.

PREREQUISITES:
    The following prerequisites are necessary to run the code. For more details, please visit the "How-to guides" link: https://aka.ms/how-to-guide

    -------Python and IDE------
    1) Install Python 3.7 or later (https://www.python.org/), which should include pip (https://pip.pypa.io/en/stable/).
    2) Install the latest version of Visual Studio Code (https://code.visualstudio.com/) or your preferred IDE. 
    
    ------Azure AI services or Document Intelligence resource------ 
    Create a single-service (https://aka.ms/single-service) or multi-service (https://aka.ms/multi-service) resource.
    You can use the free pricing tier (F0) to try the service and upgrade to a paid tier for production later.
    
    ------Get the key and endpoint------
    1) After your resource is deployed, select "Go to resource". 
    2) In the left navigation menu, select "Keys and Endpoint". 
    3) Copy one of the keys and the Endpoint for use in this sample. 
    
    ------Set your environment variables------
    At a command prompt, run the following commands, replacing <yourKey> and <yourEndpoint> with the values from your resource in the Azure portal.
    1) For Windows:
       setx DOCUMENTINTELLIGENCE_API_KEY <yourKey>
       setx DOCUMENTINTELLIGENCE_ENDPOINT <yourEndpoint>
       • You need to restart any running programs that read the environment variable.
    2) For macOS:
       export key=<yourKey>
       export endpoint=<yourEndpoint>
       • This is a temporary environment variable setting method that only lasts until you close the terminal session. 
       • To set an environment variable permanently, visit: https://aka.ms/set-environment-variables-for-macOS
    3) For Linux:
       export DOCUMENTINTELLIGENCE_API_KEY=<yourKey>
       export DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
       • This is a temporary environment variable setting method that only lasts until you close the terminal session. 
       • To set an environment variable permanently, visit: https://aka.ms/set-environment-variables-for-Linux
       
    ------Set up your programming environment------
    At a command prompt,run the following code to install the Azure AI Document Intelligence client library for Python with pip:
    pip install azure-ai-documentintelligence --pre
    
    ------Create your Python application------
    1) Create a new Python file called "sample_analyze_read.py" in an editor or IDE.
    2) Open the "sample_analyze_read.py" file and insert the provided code sample into your application.
    3) At a command prompt, use the following code to run the Python code: 
       python sample_analyze_read.py
"""

import os


def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result

# To learn the detailed concept of "span" in the following codes, visit: https://aka.ms/spans 
def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (word.span.offset + word.span.length) <= (span.offset + span.length):
            return True
    return False


def analyze_read(formUrl):
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeResult, AnalyzeDocumentRequest
    from azure.ai.documentintelligence.models import ContentFormat

    # For how to obtain the endpoint and key, please see PREREQUISITES above.
    endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
    key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
    file = os.path.basename(formUrl).split('?')[0]
    output_folder = "./data/extracted_data/"

    print("hitting", endpoint, key, "to analyze", file)

    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # Analyze a document at a URL:
    #formUrl = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/read.png"
    # Replace with your actual formUrl:
    # If you use the URL of a public website, to find more URLs, please visit: https://aka.ms/more-URLs 
    # If you analyze a document in Blob Storage, you need to generate Public SAS URL, please visit: https://aka.ms/create-sas-tokens
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(url_source=formUrl),
        output_content_format=ContentFormat.MARKDOWN,
        features=[DocumentAnalysisFeature.LANGUAGES]
    )       
    
    # # If analyzing a local document, remove the comment markers (#) at the beginning of these 11 lines.
    # # Delete or comment out the part of "Analyze a document at a URL" above.
    # # Replace <path to your sample file>  with your actual file path.
    # path_to_sample_document = "<path to your sample file>"
    # with open(path_to_sample_document, "rb") as f:
    #     poller = document_intelligence_client.begin_analyze_document(
    #         "prebuilt-read",
    #         analyze_request=f,
    #         features=[DocumentAnalysisFeature.LANGUAGES],
    #         content_type="application/octet-stream",
    #     )
    result: AnalyzeResult = poller.result()
    
    # [START analyze_read]
    # # Detect languages.
    # print("----Languages detected in the document----")
    # if result.languages is not None:
    #     for language in result.languages:
    #         print(f"Language code: '{language.locale}' with confidence {language.confidence}")
    
    # To learn the detailed concept of "bounding polygon" in the following content, visit: https://aka.ms/bounding-region
    # Analyze pages.
    for page in result.pages:
        print(f"----Analyzing document from page #{page.page_number}----")
        print(f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}")

        # Analyze lines.
        if page.lines:
            for line_idx, line in enumerate(page.lines):
                words = get_words(page, line)
                print(
                    f"...Line # {line_idx} has {len(words)} words and text '{line.content}' within bounding polygon '{line.polygon}'"
                )

                # Analyze words.
                for word in words:
                    print(f"......Word '{word.content}' has a confidence of {word.confidence}")
        
    # Analyze paragraphs.
    if result.paragraphs:
        print(f"----Detected #{len(result.paragraphs)} paragraphs in the document----")
        for paragraph in result.paragraphs:
            print(f"Found paragraph within {paragraph.bounding_regions} bounding region")
            print(f"...with content: '{paragraph.content}'")

    # write paragraphs to a file named for each file with a _paragraphs suffix:
    with open(os.path.join(output_folder, file + "_paragraphs.txt"), "w") as f:
        for paragraph in result.paragraphs:
            f.write(paragraph.content)
            f.write("\n")

    # write content in markdown format
    with open(os.path.join(output_folder, file + "_markdown.md"), "w") as f:
        f.write(result.content)
    # # Analyze tables.
    # if result.tables:
    #     print(f"----Detected #{len(result.tables)
    #     } tables in the document----") 
    #     for table_idx, table in enumerate(result.tables):




    print("----------------------------------------")
    # [END analyze_read]

if __name__ == "__main__":
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    file_url = "https://arxiv.org/pdf/2208.13773v1.pdf"

    try:
        load_dotenv('./config/.env')
        print("endpoint:", os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT"))
        print("key:", os.getenv("DOCUMENTINTELLIGENCE_API_KEY"))
        analyze_read(file_url)
    except HttpResponseError as error:
        # Examples of how to check an HttpResponseError
        # Check by error code:
        if error.error is not None:
            if error.error.code == "InvalidImage":
                print(f"Received an invalid image error: {error.error}")
            if error.error.code == "InvalidRequest":
                print(f"Received an invalid request error: {error.error}")
            # Raise the error again after printing it
            raise
        # If the inner error is None and then it is possible to check the message to get more information:
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Uh-oh! Seems there was an invalid request: {error}")
        # Raise the error again
        raise

# Next steps:
# Learn more about Layout model: https://aka.ms/di-read
# Find more sample code: https://aka.ms/doc-intelligence-samples