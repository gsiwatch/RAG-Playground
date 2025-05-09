import os
from pathlib import Path
from threading import Lock

from rag_strategies.utils.logger import setup_logger
from rag_strategies.config import settings

logger = setup_logger(__name__)

_ssl_setup_complete = False
_ssl_lock = Lock()

def setup_ssl_certificates():
    """Setup SSL certificates for the application without modifying system certs."""
    global _ssl_setup_complete
    
    with _ssl_lock:
        if _ssl_setup_complete:
            return True
        
        if not _ssl_setup_complete:
            if not settings.ssl_cert_file:
                logger.warning("SSL certificate file path not set in settings")
                return False
            
            cert_path = Path(settings.ssl_cert_file)
            
            if cert_path.exists():
                logger.info(f"Using application certificate at: {cert_path.absolute()}")
                os.environ['SSL_CERT_FILE'] = str(cert_path.absolute())
                
                if settings.requests_ca_bundle:
                    bundle_path = Path(settings.requests_ca_bundle)
                    if bundle_path.exists():
                        os.environ['REQUESTS_CA_BUNDLE'] = str(bundle_path.absolute())
                    else:
                        logger.warning(f"CA bundle file not found at: {bundle_path}")
                
                _ssl_setup_complete = True
                return True
            else:
                logger.warning(f"Certificate file not found at: {cert_path}")
                logger.warning("Run setup_certs.sh first")
                return False
        
        return True
