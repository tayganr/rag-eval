#!/bin/bash

# Fetch the PDFs from the source website
# the switches mean the following:
# -r: recursive
# -l 1: only one level deep
# -A pdf: only download pdf files
# -nd: don't create directories

mkdir -p preprocessing/source_documents_pdf
wget -r -l 1 -A pdf -nd -P ./preprocessing/source_documents_pdf https://arxiv.org/pdf/2208.13773v1.pdf

