# 测试代码 - 可以在 Jupyter notebook 中运行

from pathlib import Path
import logging
from src.textbook_indexer import TextbookIndexer


# 设置日志级别为DEBUG以查看详细信息
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_textbook_indexer():
    try:
        # 1. 首先测试 context 的构建
        context = ["Attention is", "All you need"]
        query = " ".join(context[-2:])
        print(f"构建的查询语句: {query}")
        
        # 2. 测试 textbook_indexer 的初始化
        # 假设你的 textbook_indexer 类已经导入
        textbook_path = "./test/Deep Learning Foundations and Concepts (Christopher M. Bishop, Hugh Bishop) (Z-Library).pdf"  
        textbook_indexer = TextbookIndexer(textbook_path)  # 需要替换为实际的类名
        print(f"索引器初始化完成，使用的PDF文件: {textbook_path}")
        
        # 3. 测试获取相关内容
        try:
            index_name = Path(textbook_indexer.textbook_path).stem
            print(f"使用的索引名称: {index_name}")
            
            textbook_content = textbook_indexer.get_relevant_content(
                query=query,
                index_name=index_name
            )
            
            # 4. 打印结果
            print("\n获取的内容:")
            print("-" * 50)
            print(textbook_content)
            print("-" * 50)
            
        except Exception as e:
            print(f"在获取相关内容时出错: {str(e)}")
            raise
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    test_textbook_indexer()