# ===== 彩票智能预测系统 - 容器化部署 =====
# 基础镜像：Python 3.12 slim（体积约 50MB，已含 pip 与基础编译库）
# 注：飞牛OS 默认镜像代理 docker.fnnas.com 拉 python 返回 401，改用国内可用镜像源直拉
FROM docker.m.daocloud.io/library/python:3.12-slim

# 避免交互式安装、统一编码
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# 先装依赖（利用 Docker 层缓存，改代码不会重装包）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . /app/

# 数据目录（运行时通过 volume 挂载持久化）
RUN mkdir -p /app/data

# Streamlit 默认端口
EXPOSE 8501

# 关键：绑定 0.0.0.0 才能被局域网/容器外访问
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
