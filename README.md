# 基于向量检索的知识库问答系统

这是一个基于向量检索技术的智能问答系统，支持中文语义理解，适用于文档形式的知识库管理和开放域问答。系统支持Milvus和FAISS两种向量数据库，并实现了模型加载的降级策略。

## 功能特性

- **语义理解**：使用中文BGE模型进行文本嵌入，支持自然语言理解
- **双向量数据库支持**：
  - Milvus：分布式向量数据库，适合生产环境
  - FAISS：轻量级向量索引库，适合开发测试
  - 自动降级：Milvus连接失败时自动切换到FAISS
- **模型加载策略**：优先在线加载模型，失败时从本地路径加载
- **知识库管理**：支持添加、删除、更新文档
- **Web界面**：提供直观的可视化操作界面
- **API接口**：支持通过编程方式调用系统功能
- **数据实时同步**：添加文档后无需重启即可查询

## 技术栈

- **后端框架**：Flask
- **文本嵌入**：Sentence Transformers (BAAI/bge-small-zh-v1.5)
- **向量索引**：
  - Milvus（默认，端口19530）
  - FAISS（降级方案）
- **前端**：HTML/CSS/JavaScript
- **配置管理**：python-dotenv

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑`.env`文件，根据需要修改配置：

```bash
# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_MODEL_LOCAL_PATH = "./models/bge-small-zh-v1.5"

# Vector Database Configuration
VECTOR_DB_TYPE = "milvus"  # 可选值: "milvus" 或 "faiss"

# Milvus Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION_NAME = "knowledge_base"

# FAISS Index Configuration
FAISS_INDEX_PATH = "./faiss_index.bin"

# Document Storage Configuration
DOCUMENTS_STORAGE_PATH = "./documents.json"

# Flask Web Server Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
```

### 3. 准备本地模型（可选）

如果需要本地模型加载功能，请确保模型文件存在于指定路径：

```bash
# 创建模型目录
mkdir -p ./models/bge-small-zh-v1.5
```

### 4. 运行系统

```bash
# 方式1：直接运行app.py
python app.py

# 方式2：运行main.py
python main.py
```

系统启动后，访问 http://localhost:5000 即可使用Web界面。

## 使用示例

### Web界面使用

1. **添加文档**：在"知识库管理"部分，输入文档内容和可选的元数据，点击"添加文档"按钮。
2. **提问**：在"提问"部分，输入您的问题，点击"获取答案"按钮。
3. **查看结果**：系统会显示最相关的答案和参考来源。
4. **管理文档**：在"文档列表"中可以查看和删除已添加的文档。

### API使用

#### 添加文档

```bash
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。", "metadata": {"source": "维基百科"}}'
```

#### 查询

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是人工智能？"}'
```

#### 删除文档

```bash
curl -X DELETE http://localhost:5000/api/documents/{document_id}
```

#### 获取所有文档

```bash
curl http://localhost:5000/api/documents
```

#### 获取系统统计信息

```bash
curl http://localhost:5000/api/stats
```

## 代码示例

### 通过Python代码使用

```python
from knowledge_base import KnowledgeBase
from query_processor import QueryProcessor

# 初始化知识库
kb = KnowledgeBase()

# 添加文档
doc_id = kb.add_document(
    text="机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进其性能。",
    metadata={"source": "技术文档", "category": "人工智能"}
)

print(f"添加文档成功，ID: {doc_id}")

# 初始化查询处理器（共享知识库实例）
qp = QueryProcessor(knowledge_base=kb)

# 进行查询
result = qp.process_query("什么是机器学习？")

if result["success"]:
    print(f"答案: {result['answer']}")
    print("参考来源:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. 相似度: {source['similarity']:.2f} - {source['text'][:50]}...")
```

## 知识库管理API

### POST /api/documents

添加新文档到知识库。

**请求体**：
```json
{
  "text": "文档内容",
  "metadata": {"key": "value"}
}
```

**响应**：
```json
{
  "success": true,
  "document_id": "uuid"
}
```

### DELETE /api/documents/{document_id}

删除指定ID的文档。

**响应**：
```json
{
  "success": true
}
```

### GET /api/documents

获取所有文档。

**响应**：
```json
{
  "success": true,
  "documents": [
    {
      "id": "uuid",
      "text": "文档内容",
      "metadata": {"key": "value"}
    }
  ]
}
```

### POST /api/query

提交查询并获取答案。

**请求体**：
```json
{
  "query": "用户问题"
}
```

**响应**：
```json
{
  "success": true,
  "answer": "回答内容",
  "sources": [
    {
      "id": "uuid",
      "text": "文档内容",
      "similarity": 0.95,
      "metadata": {"key": "value"}
    }
  ]
}
```

### GET /api/stats

获取系统统计信息。

**响应**：
```json
{
  "success": true,
  "stats": {
    "document_count": 10
  }
}
```

## 示例知识库

以下是一个简单的示例知识库，包含关于人工智能的基本信息：

1. **人工智能的定义**：
   "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"

2. **人工智能的历史**：
   "人工智能的概念最早可以追溯到1956年的达特茅斯会议，当时计算机科学家们首次提出了"人工智能"这一术语。"

3. **人工智能的应用**：
   "人工智能技术广泛应用于自然语言处理、计算机视觉、语音识别、推荐系统、自动驾驶等领域。"

4. **机器学习与人工智能的关系**：
   "机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进其性能，而无需明确编程。"

5. **深度学习的概念**：
   "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人类大脑的学习过程，能够处理复杂的数据模式。"

## 常见问题

### Q: 系统支持哪些语言？
A: 当前系统主要支持中文，使用的是中文BGE嵌入模型。

### Q: 可以更换嵌入模型吗？
A: 可以，只需在`.env`文件中修改`EMBEDDING_MODEL_NAME`配置即可。

### Q: 如何切换向量数据库？
A: 在`.env`文件中修改`VECTOR_DB_TYPE`配置，可选值为"milvus"或"faiss"。

### Q: 知识库的文档数量有限制吗？
A: 理论上没有限制，实际性能取决于硬件资源和所选的向量数据库。

### Q: 如何提高查询的准确性？
A: 可以尝试：
1. 使用更详细的文档内容
2. 优化文档的分段方式
3. 选择更适合的嵌入模型

### Q: Milvus连接失败怎么办？
A: 系统会自动降级到FAISS向量数据库，确保服务可用性。

## 项目结构

```
QuickBot/
├── app.py                     # Flask应用主文件
├── main.py                    # 项目入口点
├── embedding.py               # 文本嵌入模块（支持模型加载降级）
├── faiss_index.py             # FAISS向量索引实现
├── milvus_index.py            # Milvus向量索引实现
├── vector_index_factory.py    # 向量索引工厂类（支持动态切换）
├── knowledge_base.py          # 知识库管理模块
├── query_processor.py         # 查询处理模块
├── requirements.txt           # 项目依赖
├── .env                       # 环境配置
├── templates/
│   └── index.html             # Web界面模板
├── tests/
│   └── test_knowledge_base.py # 测试用例
└── README.md                  # 项目说明文档
```

## 开发与扩展

### 添加新功能

1. **更换嵌入模型**：修改`.env`文件中的`EMBEDDING_MODEL_NAME`
2. **切换向量数据库**：修改`.env`文件中的`VECTOR_DB_TYPE`
3. **优化向量检索**：修改相应的向量索引模块（faiss_index.py或milvus_index.py）
4. **添加新的API**：在`app.py`中添加新的路由
5. **改进前端界面**：修改`templates/index.html`

### 性能优化建议

1. 对于大规模知识库，建议使用Milvus向量数据库
2. 可以添加缓存机制，减少重复的文本嵌入计算
3. 考虑使用更高效的文档存储方式，如数据库
4. 对于高并发场景，可以部署多个Flask实例

## 许可证

MIT License
