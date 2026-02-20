FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY app.py tokenizer.py "${LAMBDA_TASK_ROOT}/"

CMD ["app.lambda_handler"]
