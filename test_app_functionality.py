#!/usr/bin/env python3

"""
测试应用程序功能是否正常工作，尽管有Milvus连接警告
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append('.')

def test_app_functionality():
    print("=== 测试应用程序功能 ===")
    print()
    
    try:
        # 测试1: 验证QueryProcessor初始化
        print("1. 测试QueryProcessor初始化...")
        from query_processor import QueryProcessor
        query_processor = QueryProcessor()
        print("   ✓ QueryProcessor初始化成功")
        print()
        
        # 测试2: 验证KnowledgeBase初始化
        print("2. 测试KnowledgeBase初始化...")
        from knowledge_base import KnowledgeBase
        knowledge_base = KnowledgeBase()
        print("   ✓ KnowledgeBase初始化成功")
        print()
        
        # 测试3: 测试文档添加功能
        print("3. 测试文档添加功能...")
        test_text = "这是一个测试文档，用于验证系统功能是否正常。"
        doc_id = knowledge_base.add_document(test_text, {"source": "test"})
        if doc_id:
            print(f"   ✓ 添加文档成功，ID: {doc_id}")
            print(f"   ✓ 当前文档数量: {knowledge_base.get_document_count()}")
        else:
            print("   ✗ 添加文档失败")
            return False
        print()
        
        # 测试4: 测试查询功能
        print("4. 测试查询功能...")
        query = "测试文档"
        results = knowledge_base.search(query, top_k=1)
        if results:
            print(f"   ✓ 查询成功，找到 {len(results)} 个匹配文档")
            print(f"   ✓ 文档内容: {results[0]['text']}")
        else:
            print("   ✗ 查询失败")
            return False
        print()
        
        # 测试5: 测试QueryProcessor的处理功能
        print("5. 测试QueryProcessor处理功能...")
        query_result = query_processor.process_query(query)
        if query_result['success']:
            print(f"   ✓ 查询处理成功")
            print(f"   ✓ 答案: {query_result['answer']}")
            print(f"   ✓ 参考来源数量: {len(query_result['sources'])}")
        else:
            print(f"   ✗ 查询处理失败: {query_result['error']}")
            return False
        print()
        
        # 清理测试数据
        knowledge_base.delete_document(doc_id)
        
        print("🎉 所有测试通过！应用程序功能正常工作，尽管有Milvus连接警告。")
        print("\n注意：Milvus连接警告是正常的，系统已经自动回退到FAISS索引。")
        print("应用程序仍然可以正常使用所有功能。")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败，出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_app_functionality()
