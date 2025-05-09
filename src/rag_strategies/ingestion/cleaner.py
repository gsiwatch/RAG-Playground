import re
from bs4 import BeautifulSoup
from rag_strategies.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContentCleaner:
    def clean_content(self, content: str) -> str:
        """Clean content by removing HTML tags and normalizing text"""
        try:
            # Remove HTML tags
            cleaned = self._remove_html_safely(content)
            
            # Normalize whitespace
            cleaned = self._normalize_whitespace(cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"Error in content cleaning: {str(e)}")
            return self._fallback_cleaning(content)

    def _remove_html_safely(self, content: str) -> str:
        """Remove HTML while preserving text"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n')
            return text
            
        except Exception as e:
            logger.error(f"Error removing HTML: {str(e)}")
            return content

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        try:
            # Normalize line endings
            text = text.replace('\r\n', '\n')
            
            # Remove extra spaces
            text = re.sub(r' +', ' ', text)
            
            # Normalize multiple newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing whitespace: {str(e)}")
            return ' '.join(text.split())

    def _fallback_cleaning(self, content: str) -> str:
        """Fallback cleaning method"""
        try:
            # Basic HTML removal
            text = BeautifulSoup(content, 'html.parser').get_text()
            
            # Basic whitespace normalization
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in fallback cleaning: {str(e)}")
            return content.strip()
        
