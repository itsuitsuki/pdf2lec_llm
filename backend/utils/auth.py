from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置常量
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")  # 在生产环境中必须使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/v1/login",
    auto_error=True,
    scheme_name="Bearer"
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    :param plain_password: 明文密码
    :param hashed_password: 哈希后的密码
    :return: 是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    获取密码的哈希值
    :param password: 明文密码
    :return: 哈希后的密码
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建访问令牌
    :param data: 要编码的数据
    :param expires_delta: 过期时间增量
    :return: JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    获取当前用户
    :param token: JWT token
    :return: 用户邮箱
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 解码 JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        
        # 检查token是否过期
        exp = payload.get("exp")
        if exp is None:
            raise credentials_exception
        
        if datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return email
        
    except JWTError:
        raise credentials_exception

def create_refresh_token(data: dict) -> str:
    """
    创建刷新令牌（可选功能）
    :param data: 要编码的数据
    :return: JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # 刷新令牌有效期7天
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """
    验证令牌
    :param token: JWT token
    :return: 解码后的数据或None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

class AuthError(Exception):
    """
    认证错误基类
    """
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class InvalidTokenError(AuthError):
    """
    无效token错误
    """
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, 401)

class ExpiredTokenError(AuthError):
    """
    token过期错误
    """
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, 401)

class InvalidCredentialsError(AuthError):
    """
    无效凭证错误
    """
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message, 401)