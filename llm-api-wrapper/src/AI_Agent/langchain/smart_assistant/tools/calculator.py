from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式，如 "2+2"、"(10*5)+20"、"2**8"
    """
    try:
        # 安全检查
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression.replace('**', '')):
            return "表达式包含非法字符"
        
        result = eval(expression, {"__builtins__": {}}, {})
        
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        
        return f"计算结果：{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{str(e)}"