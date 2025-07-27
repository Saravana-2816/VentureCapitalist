from crawl4ai import AsyncWebCrawler
import asyncio

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://instagram.com")
        print(result.media["images"])  # List of image URLs
        print(result.media["videos"])  # List of video URLs
        print(result.media["audio"])   # List of audio URLs

asyncio.run(main())