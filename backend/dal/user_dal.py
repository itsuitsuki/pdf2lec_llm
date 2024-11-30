import json
from typing import Optional, Dict
from redis import Redis
import logging

logger = logging.getLogger(__name__)

class UserDAL:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.user_key_prefix = "user:"
    
    def _get_user_key(self, email: str) -> str:
        return f"{self.user_key_prefix}{email}"
    
    def get_user(self, email: str) -> Optional[Dict]:
        try:
            user_data = self.redis_client.get(self._get_user_key(email))
            return json.loads(user_data) if user_data else None
        except Exception as e:
            logger.error(f"Error getting user {email}: {str(e)}")
            return None
    
    def create_user(self, user_data: dict) -> bool:
        try:
            # 使用 setnx 确保原子性操作，只有当 key 不存在时才设置
            key = self._get_user_key(user_data["email"])
            success = self.redis_client.setnx(key, json.dumps(user_data))
            if success:
                # 确保数据被持久化
                self.redis_client.save()
            return success
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def update_user(self, email: str, user_data: dict) -> bool:
        try:
            self.redis_client.set(
                self._get_user_key(email),
                json.dumps(user_data)
            )
            # 确保数据被持久化
            self.redis_client.save()
            return True
        except Exception as e:
            logger.error(f"Error updating user {email}: {str(e)}")
            return False
    
    def delete_user(self, email: str) -> bool:
        try:
            result = self.redis_client.delete(self._get_user_key(email))
            # 确保数据被持久化
            self.redis_client.save()
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting user {email}: {str(e)}")
            return False
    
    def add_pdf_access(self, email: str, pdf_id: str) -> bool:
        try:
            user_data = self.get_user(email)
            if user_data:
                if 'accessible_pdfs' not in user_data:
                    user_data['accessible_pdfs'] = []
                if pdf_id not in user_data['accessible_pdfs']:
                    user_data['accessible_pdfs'].append(pdf_id)
                    return self.update_user(email, user_data)
            return False
        except Exception as e:
            logger.error(f"Error adding PDF access for user {email}: {str(e)}")
            return False