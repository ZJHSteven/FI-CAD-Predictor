# -*- coding: utf-8 -*-
# 本文件集中处理API的异常与错误响应。
# 这样可以保证主流程代码更清晰，也便于统一日志与错误格式。

from typing import Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """
    自定义API错误，用于业务层主动抛出可控异常。

    Attributes:
        message: 错误说明
        status_code: HTTP状态码
        details: 可选的细节信息
    """

    def __init__(self, message: str, status_code: int = 400, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """
    ApiError 的统一处理器。

    Args:
        request: FastAPI 请求对象
        exc: 自定义ApiError异常

    Returns:
        统一结构的JSON响应
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
        },
    )
