# FastAPI Project

这是一个使用FastAPI框架构建的项目模板。

## 项目结构

```
├── app/
│   ├── api/            # API路由
│   ├── core/           # 核心配置
│   ├── db/             # 数据库配置
│   ├── models/         # SQLAlchemy模型
│   └── schemas/        # Pydantic模型
├── .env                # 环境变量配置
├── .env.example        # 环境变量示例
├── main.py            # 应用入口
└── requirements.txt   # 项目依赖
```

## 安装

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
- 复制 `.env.example` 到 `.env`
- 修改 `.env` 中的配置

## 运行

启动开发服务器：
```bash
uvicorn main:app --reload
```

访问 API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 