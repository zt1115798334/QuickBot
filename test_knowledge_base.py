#!/usr/bin/env python3

"""
知识库全面测试用例
支持Milvus和FAISS两种向量数据库
"""

import sys
import os
import time
import uuid

# 添加项目根目录到Python路径
sys.path.append('.')

from knowledge_base import KnowledgeBase

def test_single_document():
    """测试单个文档的添加和查询"""
    print("\n=== 测试1: 单个文档添加和查询 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 添加单个文档
    doc_text = "这是一个测试文档，用于测试知识库的基本功能。"
    doc_metadata = {"source": "test", "category": "basic"}
    doc_id = kb.add_document(text=doc_text, metadata=doc_metadata)
    
    print(f"✓ 添加文档成功，ID: {doc_id}")
    print(f"✓ 当前文档数量: {kb.get_document_count()}")
    
    # 查询文档
    query = "测试文档"
    results = kb.search(query, top_k=1)
    
    if results:
        print(f"✓ 查询成功，找到匹配文档")
        print(f"  文档内容: {results[0]['text']}")
        print(f"  相似度: {results[0]['similarity']}")
        print(f"  元数据: {results[0]['metadata']}")
    else:
        print(f"✗ 查询失败，未找到匹配文档")
        return False
    
    return True

def test_multiple_documents():
    """测试多个文档的添加和批量查询"""
    print("\n=== 测试2: 多个文档添加和查询 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 准备测试数据
    test_docs = [
        {"text": "Python是一种广泛使用的解释型、高级和通用的编程语言。", "metadata": {"source": "wiki", "language": "zh"}},
        {"text": "JavaScript是一种具有函数优先的轻量级，解释型或即时编译型的编程语言。", "metadata": {"source": "wiki", "language": "zh"}},
        {"text": "Java是一种广泛使用的计算机编程语言，拥有跨平台、面向对象、泛型编程的特性。", "metadata": {"source": "wiki", "language": "zh"}},
        {"text": "C++是C语言的继承，它既可以进行C语言的过程化程序设计，又可以进行以抽象数据类型为特点的基于对象的程序设计。", "metadata": {"source": "wiki", "language": "zh"}}
    ]
    
    # 批量添加文档
    texts = [doc["text"] for doc in test_docs]
    metadatas = [doc["metadata"] for doc in test_docs]
    doc_ids = kb.add_documents(texts, metadatas)
    
    print(f"✓ 批量添加文档成功，共添加 {len(doc_ids)} 个文档")
    print(f"✓ 当前文档数量: {kb.get_document_count()}")
    
    # 查询不同主题
    queries = [
        ("Python", "Python相关文档"),
        ("JavaScript", "JavaScript相关文档"),
        ("编程语言", "所有编程语言文档")
    ]
    
    for query, desc in queries:
        print(f"\n  查询: {query} ({desc})")
        results = kb.search(query, top_k=2)
        for i, result in enumerate(results):
            print(f"    {i+1}. {result['text'][:50]}... (相似度: {result['similarity']:.4f})")
    
    return True

def test_document_update_delete():
    """测试文档的更新和删除"""
    print("\n=== 测试3: 文档更新和删除 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 添加文档
    original_text = "这是原始文档内容。"
    doc_id = kb.add_document(original_text, {"version": "1.0"})
    print(f"✓ 添加原始文档成功，ID: {doc_id}")
    
    # 更新文档
    updated_text = "这是更新后的文档内容。"
    update_result = kb.update_document(doc_id, updated_text, {"version": "2.0"})
    print(f"✓ 更新文档结果: {update_result}")
    
    # 查询验证更新
    results = kb.search("更新后的文档", top_k=1)
    if results and "更新后的文档内容" in results[0]["text"]:
        print("✓ 文档更新验证成功")
    else:
        print("✗ 文档更新验证失败")
        return False
    
    # 删除文档
    delete_result = kb.delete_document(doc_id)
    print(f"✓ 删除文档结果: {delete_result}")
    print(f"✓ 删除后文档数量: {kb.get_document_count()}")
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试4: 边界情况测试 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 测试1: 添加空文档
    empty_doc_id = kb.add_document("", {"test": "empty"})
    print(f"✓ 添加空文档结果: {empty_doc_id} (预期: None)")
    
    # 测试2: 添加非常长的文档
    long_text = "测试" * 1000  # 4000字符
    long_doc_id = kb.add_document(long_text, {"test": "long_text"})
    print(f"✓ 添加长文档结果: {long_doc_id is not None}")
    
    # 测试3: 空查询
    empty_query_results = kb.search("", top_k=1)
    print(f"✓ 空查询结果数量: {len(empty_query_results)} (预期: 0)")
    
    # 测试4: 删除不存在的文档
    non_exist_id = str(uuid.uuid4())
    delete_result = kb.delete_document(non_exist_id)
    print(f"✓ 删除不存在文档结果: {delete_result} (预期: False)")
    
    # 测试5: 更新不存在的文档
    update_result = kb.update_document(non_exist_id, "new text")
    print(f"✓ 更新不存在文档结果: {update_result} (预期: False)")
    
    return True

def test_performance():
    """测试基本性能"""
    print("\n=== 测试5: 基本性能测试 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 准备100个测试文档
    num_docs = 100
    test_docs = []
    for i in range(num_docs):
        test_docs.append({
            "text": f"这是性能测试文档 #{i}，包含一些示例内容来测试知识库的性能。",
            "metadata": {"test": "performance", "index": i}
        })
    
    # 批量添加文档
    start_time = time.time()
    texts = [doc["text"] for doc in test_docs]
    metadatas = [doc["metadata"] for doc in test_docs]
    doc_ids = kb.add_documents(texts, metadatas)
    end_time = time.time()
    
    print(f"✓ 添加 {len(doc_ids)} 个文档耗时: {end_time - start_time:.2f} 秒")
    print(f"✓ 当前文档数量: {kb.get_document_count()}")
    
    # 测试不同top_k的查询性能
    for top_k in [1, 5, 10, 20]:
        start_time = time.time()
        results = kb.search("性能测试文档", top_k=top_k)
        end_time = time.time()
        print(f"  查询top_k={top_k} 耗时: {end_time - start_time:.4f} 秒, 结果数量: {len(results)}")
    
    return True

def test_metadata_handling():
    """测试元数据处理"""
    print("\n=== 测试6: 元数据处理测试 ===")
    
    kb = KnowledgeBase()
    kb.clear_all()
    
    # 添加带不同元数据的文档
    metadata_tests = [
        {"text": "文档1内容", "metadata": {"author": "张三", "category": "技术", "tags": ["python", "编程"]}},
        {"text": "文档2内容", "metadata": {"author": "李四", "category": "技术", "tags": ["java", "编程"]}},
        {"text": "文档3内容", "metadata": {"author": "王五", "category": "文学", "tags": ["小说", "阅读"]}},
    ]
    
    for doc in metadata_tests:
        doc_id = kb.add_document(doc["text"], doc["metadata"])
        print(f"✓ 添加带元数据的文档成功，ID: {doc_id}")
    
    print(f"✓ 当前文档数量: {kb.get_document_count()}")
    
    # 查询并验证元数据
    query = "文档内容"
    results = kb.search(query, top_k=3)
    
    for i, result in enumerate(results):
        print(f"\n  查询结果 {i+1}:")
        print(f"    文本: {result['text']}")
        print(f"    相似度: {result['similarity']:.4f}")
        print(f"    元数据: {result['metadata']}")
    
    return True

def run_all_tests():
    """运行所有测试"""
    print("知识库全面测试开始")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_single_document,
        test_multiple_documents,
        test_document_update_delete,
        test_edge_cases,
        test_performance,
        test_metadata_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"\n{test.__name__} ✓ 测试通过")
            else:
                failed += 1
                print(f"\n{test.__name__} ✗ 测试失败")
        except Exception as e:
            failed += 1
            print(f"\n{test.__name__} ✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== 测试结果汇总 ===")
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ 有 {failed} 个测试失败")
        return 1

if __name__ == "__main__":
    # 运行所有测试
    sys.exit(run_all_tests())
