import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("INGESTOR")

class SpaceWeatherIngestor:
    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        self.xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
    
    async def _fetch_json(self, session: aiohttp.ClientSession, url: str) -> Optional[Any]:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except asyncio.TimeoutError:
            logger.error(f"ERR_TIMEOUT | URL={url} | LIMIT={self.timeout.total}s")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"ERR_NETWORK | URL={url} | DETAILS={e}")
            return None

    async def fetch_kp_index(self, session: aiohttp.ClientSession) -> float:
        data = await self._fetch_json(session, self.kp_url)
        # NOAA API ilk satırda header döndüğü için len < 2 kontrolü zorunludur
        if not data or len(data) < 2:
            return 0.0 
        
        try:
            latest_entry = data[-1]
            if isinstance(latest_entry, list):
                return float(latest_entry[1])
            elif isinstance(latest_entry, dict):
                kp_val = latest_entry.get('Kp') or latest_entry.get('kp_index') or latest_entry.get('kp')
                return float(kp_val) if kp_val is not None else 0.0
            return 0.0
        except (IndexError, ValueError, KeyError, TypeError) as e:
            logger.error(f"ERR_PARSE_KP | DETAILS={e}")
            return 0.0

    async def fetch_xray_flux(self, session: aiohttp.ClientSession) -> float:
        data = await self._fetch_json(session, self.xray_url)
        if not data:
            return 1e-8 
        
        try:
            latest_flux = float(data[-1].get('flux', 1e-8))
            return latest_flux
        except (IndexError, ValueError, KeyError, TypeError) as e:
            logger.error(f"ERR_PARSE_XRAY | DETAILS={e}")
            return 1e-8

    async def get_current_context(self) -> Dict[str, float]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            kp_val, xray_val = await asyncio.gather(
                self.fetch_kp_index(session),
                self.fetch_xray_flux(session)
            )
            
            return {
                "kp_index": kp_val,
                "xray_flux": xray_val
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s')
    
    async def _test():
        ingestor = SpaceWeatherIngestor()
        logger.info("STATE: FETCHING_TELEMETRY | TARGET: NOAA_SWPC")
        context = await ingestor.get_current_context()
        logger.info(f"DATA_FETCHED | KP={context['kp_index']:.2f} XRAY={context['xray_flux']:.2e}")
        
    asyncio.run(_test())