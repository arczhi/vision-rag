FROM --platform=linux/amd64 python:3.11-slim

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        datasette==0.65.1

COPY docker/sqlite-ui/run-datasette.sh /usr/local/bin/run-datasette
RUN chmod +x /usr/local/bin/run-datasette

WORKDIR /data
EXPOSE 8001

CMD ["run-datasette"]
