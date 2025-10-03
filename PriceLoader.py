import os
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SLICKCHARTS_SP500_URL = "https://www.slickcharts.com/sp500"

def _to_yahoo_symbol(ticker: str) -> str:
    # BRK.B -> BRK-B, BF.B -> BF-B, etc.
    return ticker.replace('.', '-').upper().strip()

class PriceLoader:
    def __init__(self, data_dir='data/prices', start_date='2005-01-01', end_date='2025-01-01', batch_size=15):
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.tickers = self._get_sp500_tickers_from_slickcharts()
        # 统一成 Yahoo 可用的代码
        self.yf_tickers = [_to_yahoo_symbol(t) for t in self.tickers]
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_sp500_tickers_from_slickcharts(self):
        logging.info(f"Scraping S&P 500 tickers from {SLICKCHARTS_SP500_URL}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.slickcharts.com/"
        }
        try:
            request = Request(SLICKCHARTS_SP500_URL, headers=headers)
            with urlopen(request, timeout=15) as resp:
                tables = pd.read_html(resp.read())
                sp500_df = tables[0]
                tickers = sp500_df["Symbol"].dropna().astype(str).str.upper().str.strip().tolist()
                if 400 <= len(tickers) <= 600:
                    logging.info(f"Scraped {len(tickers)} S&P 500 tickers (SlickCharts).")
                    return tickers
                logging.warning(f"Unexpected ticker count ({len(tickers)}). Using fallback list.")
        except (HTTPError, URLError, Exception) as e:
            logging.error(f"SlickCharts scrape failed: {str(e)}. Using fallback list.")

        return [
            'NVDA','MSFT','AAPL','AMZN','META','AVGO','GOOGL','GOOG','BRK.B','TSLA',
            'JPM','WMT','LLY','V','ORCL','MA','NFLX','XOM','COST','JNJ','HD','PG',
            'PLTR','BAC','ABBV','CVX','KO','GE','TMUS','CSCO','WFC','UNH','AMD','PM'
        ]

    def _download_single_ticker(self, yf_ticker):
        """单票下载：统一 auto_adjust=False + 取 Adj Close。"""
        try:
            data = yf.download(
                yf_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,   # 关键：这样一定有 'Adj Close'
                progress=False,
            )
            if not data.empty:
                if 'Adj Close' in data.columns:
                    return data['Adj Close'].dropna()
                elif 'Close' in data.columns:
                    # 兜底（极少数情况）
                    return data['Close'].dropna()
            logging.warning(f"No adjusted close data for {yf_ticker}.")
        except Exception as e:
            logging.error(f"Single-ticker download failed for {yf_ticker}: {str(e)}")
        return pd.Series(dtype='float64')

    def _download_batch(self, yf_batch):
        """批量下载 + 单票补漏（健壮提取 Adj Close，无论列层级顺序如何）."""
        def _extract_adj_close_matrix(raw: pd.DataFrame) -> pd.DataFrame:
            if raw is None or raw.empty:
                return pd.DataFrame()
            # 多重列：尝试在任意层级切 'Adj Close'
            if isinstance(raw.columns, pd.MultiIndex):
                for lvl in range(raw.columns.nlevels):
                    try:
                        mat = raw.xs('Adj Close', axis=1, level=lvl, drop_level=True)
                        # 保证是 DataFrame
                        if isinstance(mat, pd.Series):
                            mat = mat.to_frame()
                        return mat
                    except KeyError:
                        continue
                # 兜底再试 'Close'
                for lvl in range(raw.columns.nlevels):
                    try:
                        mat = raw.xs('Close', axis=1, level=lvl, drop_level=True)
                        if isinstance(mat, pd.Series):
                            mat = mat.to_frame()
                        return mat
                    except KeyError:
                        continue
                return pd.DataFrame()
            # 单层列：单票情况
            else:
                if 'Adj Close' in raw.columns:
                    # 单票时列名不是 ticker，所以重命名成该票名，便于统一拼接
                    return raw[['Adj Close']].rename(columns={'Adj Close': yf_batch[0]})
                if 'Close' in raw.columns:  # 兜底
                    return raw[['Close']].rename(columns={'Close': yf_batch[0]})
                return pd.DataFrame()

        batch_data = pd.DataFrame()
        try:
            raw = yf.download(
                yf_batch,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,   # 保证应该有 Adj Close
                progress=False,
                group_by='ticker',
                threads=True,
            )
            batch_data = _extract_adj_close_matrix(raw)
        except Exception as e:
            logging.warning(f"Batch download failed (transport): {str(e)}. Will fallback to single-ticker.")

        # 用单票补缺（包括 batch_data 为空或缺列）
        missing = [t for t in yf_batch if (batch_data.empty or t not in batch_data.columns)]
        if missing:
            logging.info(f"Fetching missing tickers individually ({len(missing)}): "
                        f"{', '.join(missing[:8])}{'...' if len(missing)>8 else ''}")
            for t in missing:
                s = self._download_single_ticker(t)
                if not s.empty:
                    batch_data[t] = s
                time.sleep(0.4)

        return batch_data


    def download_all(self, force_redownload=False, delay_seconds=1.2):
        logging.info("Starting S&P 500 price download (Adj Close, 2005–2025)...")
        total = len(self.yf_tickers)
        downloaded, skipped_existing, skipped_short = 0, 0, 0

        with tqdm(total=total, desc="Total Tickers Processed") as pbar:
            for i in range(0, total, self.batch_size):
                original_batch = self.yf_tickers[i:i+self.batch_size]
                existing = []
                need_fetch = original_batch
                if not force_redownload:
                    existing = [t for t in original_batch if os.path.exists(os.path.join(self.data_dir, f"{t}.parquet"))]
                    need_fetch = [t for t in original_batch if t not in existing]
                    skipped_existing += len(existing)
                    pbar.update(len(existing))  # 正确更新进度

                if not need_fetch:
                    time.sleep(0.2)
                    continue

                batch_df = self._download_batch(need_fetch)
                if batch_df.empty:
                    logging.error(f"Batch {i//self.batch_size + 1} failed entirely. Skipping.")
                    pbar.update(len(need_fetch))
                    time.sleep(delay_seconds * 2)
                    continue

                for t in need_fetch:
                    if t not in batch_df.columns:
                        logging.warning(f"{t} still missing after fallback. Skipping.")
                        pbar.update(1)
                        continue

                    s = batch_df[t].dropna()
                    if len(s) < 252:  # 少于一年交易日，跳过
                        skipped_short += 1
                        pbar.update(1)
                        continue

                    df_out = s.to_frame(name='adj_close')
                    fp = os.path.join(self.data_dir, f"{t}.parquet")
                    df_out.to_parquet(fp)
                    downloaded += 1
                    pbar.update(1)

                time.sleep(delay_seconds)

        logging.info("Download complete!")
        logging.info(f"  - Saved      : {downloaded}")
        logging.info(f"  - Skipped old: {skipped_existing}")
        logging.info(f"  - Skipped <1y: {skipped_short}")

    def load_price(self, ticker):
        """加载时也允许传 BRK.B / BRK-B，统一转成 Yahoo 格式寻找文件。"""
        t = _to_yahoo_symbol(ticker)
        fp = os.path.join(self.data_dir, f"{t}.parquet")
        if os.path.exists(fp):
            try:
                df = pd.read_parquet(fp)
                df.index = pd.to_datetime(df.index)
                return df.sort_index()
            except Exception as e:
                logging.error(f"Load failed for {ticker}: {str(e)}")
        else:
            logging.warning(f"No data file for {ticker} at {fp}")
        return None

    def get_available_tickers(self):
        return sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith('.parquet')])

if __name__ == '__main__':
    loader = PriceLoader(
        data_dir='/data',
        start_date='2005-01-01',
        end_date='2025-01-01',
        batch_size=15
    )
    loader.download_all(force_redownload=False, delay_seconds=1.2)
    print(f"Total cached: {len(loader.get_available_tickers())}")
