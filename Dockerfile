# This docker file is used in jenkins to automatically build the documentation once a day
FROM python:3.8 AS docs_builder

# Copy source code and install library
COPY setup.py /tiger/lib/setup.py
COPY tiger/ /tiger/lib/tiger/
RUN pip install --no-cache-dir --editable /tiger/lib/.[docs]

# Build documentation
COPY docs/ /tiger/docs/src/
RUN mkdir /tiger/docs/build
RUN sphinx-build -b html -d "/tiger/docs/doctrees" "/tiger/docs/src" "/tiger/docs/build"
