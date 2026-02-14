"""
Portfolio Optimizer - AI-Powered Document Parser
================================================

Extracts portfolio holdings, transactions, and historical data from
uploaded PDF statements and CSV/Excel files using AI and traditional parsing.

Supported Formats:
- PDF: Brokerage statements, trade confirmations, portfolio summaries
- CSV: Transaction exports, position reports
- Excel: Multi-sheet portfolio reports
"""

import os
import re
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import io

import pandas as pd
import numpy as np

# PDF processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Excel processing
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from services.ai_insights import LLMClient

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    PDF_STATEMENT = "pdf_statement"
    CSV_EXPORT = "csv_export"
    EXCEL_REPORT = "excel_report"
    UNKNOWN = "unknown"


class BrokerType(Enum):
    """Common brokerage platforms for format hints."""
    FIDELITY = "fidelity"
    SCHWAB = "schwab"
    VANGUARD = "vanguard"
    IBKR = "interactive_brokers"
    ETRADE = "etrade"
    TD_AMERITRADE = "td_ameritrade"
    GENERIC = "generic"


@dataclass
class ExtractedHolding:
    """
    Single holding extracted from document.
    """
    ticker: str
    name: Optional[str] = None
    shares: Optional[float] = None
    price: Optional[float] = None
    market_value: Optional[float] = None
    cost_basis: Optional[float] = None
    purchase_date: Optional[date] = None
    asset_class: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ExtractedTransaction:
    """
    Single transaction extracted from document.
    """
    date: date
    ticker: str
    transaction_type: str
    shares: Optional[float] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    fees: Optional[float] = 0.0


@dataclass
class ParsedPortfolio:
    """
    Complete portfolio extracted from document.
    """
    holdings: List[ExtractedHolding] = field(default_factory=list)
    transactions: List[ExtractedTransaction] = field(default_factory=list)
    cash_balance: Optional[float] = None
    account_value: Optional[float] = None
    statement_date: Optional[date] = None
    broker: Optional[str] = None
    extraction_confidence: float = 0.0
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'holdings': [
                {
                    'ticker': h.ticker,
                    'name': h.name,
                    'shares': h.shares,
                    'price': h.price,
                    'market_value': h.market_value,
                    'cost_basis': h.cost_basis,
                    'purchase_date': h.purchase_date.isoformat() if h.purchase_date else None,
                    'asset_class': h.asset_class,
                    'confidence': h.confidence
                }
                for h in self.holdings
            ],
            'transactions': [
                {
                    'date': t.date.isoformat(),
                    'ticker': t.ticker,
                    'transaction_type': t.transaction_type,
                    'shares': t.shares,
                    'price': t.price,
                    'amount': t.amount,
                    'fees': t.fees
                }
                for t in self.transactions
            ],
            'cash_balance': self.cash_balance,
            'account_value': self.account_value,
            'statement_date': self.statement_date.isoformat() if self.statement_date else None,
            'broker': self.broker,
            'extraction_confidence': self.extraction_confidence
        }


class DocumentParser:
    """
    Main document parsing service with AI-powered extraction.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def parse(
        self,
        file_bytes: bytes,
        filename: str,
        broker_hint: Optional[str] = None
    ) -> ParsedPortfolio:
        """
        Parse document and extract portfolio information.
        """
        doc_type = self._detect_document_type(filename)
        
        logger.info(f"Parsing {doc_type.value} document: {filename}")
        
        if doc_type == DocumentType.PDF_STATEMENT:
            return self._parse_pdf(file_bytes, broker_hint)
        elif doc_type == DocumentType.CSV_EXPORT:
            return self._parse_csv(file_bytes, broker_hint)
        elif doc_type == DocumentType.EXCEL_REPORT:
            return self._parse_excel(file_bytes, broker_hint)
        else:
            raise ValueError(f"Unsupported document type: {filename}")
    
    def _detect_document_type(self, filename: str) -> DocumentType:
        """Detect document type from filename extension."""
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            return DocumentType.PDF_STATEMENT
        elif ext in ['csv', 'txt']:
            return DocumentType.CSV_EXPORT
        elif ext in ['xlsx', 'xls']:
            return DocumentType.EXCEL_REPORT
        else:
            return DocumentType.UNKNOWN
    
    def _parse_pdf(
        self,
        file_bytes: bytes,
        broker_hint: Optional[str] = None
    ) -> ParsedPortfolio:
        """
        Parse PDF document using pdfplumber + LLM extraction.
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required for PDF parsing")
        
        all_text = []
        tables = []
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
                
                page_tables = page.extract_tables()
                tables.extend(page_tables)
        
        combined_text = "\n\n".join(all_text)
        
        # Try to find holdings table in extracted tables first
        holdings = self._extract_holdings_from_tables(tables)
        
        # If table extraction didn't work well, use LLM
        if len(holdings) < 2:
            holdings = self._extract_holdings_with_llm(combined_text)
        
        # Extract metadata
        statement_date = self._extract_date(combined_text)
        broker = broker_hint or self._detect_broker(combined_text)
        cash_balance = self._extract_cash_balance(combined_text)
        account_value = self._extract_account_value(combined_text)
        
        # Calculate confidence
        avg_confidence = np.mean([h.confidence for h in holdings]) if holdings else 0.0
        
        return ParsedPortfolio(
            holdings=holdings,
            cash_balance=cash_balance,
            account_value=account_value,
            statement_date=statement_date,
            broker=broker,
            extraction_confidence=avg_confidence,
            raw_text=combined_text[:5000] if combined_text else None
        )
    
    def _parse_csv(
        self,
        file_bytes: bytes,
        broker_hint: Optional[str] = None
    ) -> ParsedPortfolio:
        """
        Parse CSV/TSV export using pandas + LLM column mapping.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Could not decode CSV file")
        
        # Use LLM to identify column types
        column_mapping = self._map_columns_with_llm(df.columns.tolist())
        
        holdings = []
        transactions = []
        
        # Extract holdings if we have position data
        if 'ticker' in column_mapping:
            ticker_col = column_mapping['ticker']
            shares_col = column_mapping.get('shares')
            price_col = column_mapping.get('price')
            value_col = column_mapping.get('market_value')
            
            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).strip().upper() if pd.notna(row[ticker_col]) else None
                if ticker and len(ticker) <= 5 and ticker.isalpha():
                    holding = ExtractedHolding(
                        ticker=ticker,
                        name=str(row.get(column_mapping.get('name'), '')) if 'name' in column_mapping else None,
                        shares=float(row[shares_col]) if shares_col and pd.notna(row[shares_col]) else None,
                        price=float(row[price_col]) if price_col and pd.notna(row[price_col]) else None,
                        market_value=float(row[value_col]) if value_col and pd.notna(row[value_col]) else None,
                        confidence=0.9
                    )
                    holdings.append(holding)
        
        return ParsedPortfolio(
            holdings=holdings,
            transactions=transactions,
            extraction_confidence=0.9 if holdings else 0.3,
            broker=broker_hint
        )
    
    def _parse_excel(
        self,
        file_bytes: bytes,
        broker_hint: Optional[str] = None
    ) -> ParsedPortfolio:
        """Parse Excel file with multiple sheets."""
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel parsing")
        
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        all_holdings = []
        
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            if len(df) > 0:
                sheet_result = self._parse_dataframe(df, broker_hint)
                all_holdings.extend(sheet_result.holdings)
        
        return ParsedPortfolio(
            holdings=all_holdings,
            broker=broker_hint,
            extraction_confidence=0.85 if all_holdings else 0.3
        )
    
    def _parse_dataframe(
        self,
        df: pd.DataFrame,
        broker_hint: Optional[str] = None
    ) -> ParsedPortfolio:
        """Parse a DataFrame to extract holdings."""
        column_mapping = self._map_columns_with_llm(df.columns.tolist())
        holdings = []
        
        if 'ticker' in column_mapping:
            ticker_col = column_mapping['ticker']
            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).strip().upper() if pd.notna(row[ticker_col]) else None
                if ticker and len(ticker) <= 5:
                    holdings.append(ExtractedHolding(
                        ticker=ticker,
                        shares=float(row[column_mapping['shares']]) if 'shares' in column_mapping and pd.notna(row[column_mapping['shares']]) else None,
                        confidence=0.9
                    ))
        
        return ParsedPortfolio(holdings=holdings)
    
    def _extract_holdings_from_tables(
        self,
        tables: List[List[List[str]]]
    ) -> List[ExtractedHolding]:
        """Extract holdings from PDF tables."""
        holdings = []
        
        for table in tables:
            if not table or len(table) < 2:
                continue
            
            header = [str(cell).lower() if cell else '' for cell in table[0]]
            
            ticker_idx = None
            shares_idx = None
            price_idx = None
            value_idx = None
            
            for i, col in enumerate(header):
                if any(word in col for word in ['symbol', 'ticker', 'stock']):
                    ticker_idx = i
                elif any(word in col for word in ['quantity', 'shares', 'qty']):
                    shares_idx = i
                elif any(word in col for word in ['price', 'last price']):
                    price_idx = i
                elif any(word in col for word in ['value', 'market value', 'mv']):
                    value_idx = i
            
            if ticker_idx is not None:
                for row in table[1:]:
                    if len(row) > ticker_idx:
                        ticker = str(row[ticker_idx]).strip().upper()
                        if ticker and len(ticker) <= 5 and ticker.isalpha():
                            holding = ExtractedHolding(
                                ticker=ticker,
                                shares=float(row[shares_idx]) if shares_idx and row[shares_idx] else None,
                                price=float(row[price_idx]) if price_idx and row[price_idx] else None,
                                market_value=float(row[value_idx]) if value_idx and row[value_idx] else None,
                                confidence=0.8
                            )
                            holdings.append(holding)
        
        return holdings
    
    def _extract_holdings_with_llm(self, text: str) -> List[ExtractedHolding]:
        """Use LLM to extract holdings from unstructured text."""
        prompt = """Extract all stock holdings from this brokerage statement.

Text from statement:
""" + text[:8000] + """

Extract holdings in this JSON format:
{
  "holdings": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc",
      "shares": 100,
      "price": 150.00,
      "market_value": 15000.00,
      "cost_basis": 140.00,
      "purchase_date": "2023-01-15",
      "asset_class": "Equity"
    }
  ]
}

Only include actual stock/ETF holdings, not cash or other positions. If a field is not available, use null."""
        
        try:
            response = self.llm.complete(prompt, temperature=0.1)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                holdings = []
                for h in data.get('holdings', []):
                    purchase_date = None
                    if h.get('purchase_date'):
                        try:
                            purchase_date = datetime.strptime(h['purchase_date'], '%Y-%m-%d').date()
                        except ValueError:
                            pass
                    
                    holding = ExtractedHolding(
                        ticker=h.get('ticker', 'UNKNOWN'),
                        name=h.get('name'),
                        shares=float(h['shares']) if h.get('shares') else None,
                        price=float(h['price']) if h.get('price') else None,
                        market_value=float(h['market_value']) if h.get('market_value') else None,
                        cost_basis=float(h['cost_basis']) if h.get('cost_basis') else None,
                        purchase_date=purchase_date,
                        asset_class=h.get('asset_class'),
                        confidence=0.85
                    )
                    holdings.append(holding)
                
                return holdings
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
        
        return []
    
    def _map_columns_with_llm(self, columns: List[str]) -> Dict[str, str]:
        """Use LLM to map column names to standard schema."""
        columns_json = json.dumps(columns)
        prompt = f"""Map these CSV column names to standard portfolio fields:

Columns: {columns_json}

Return a JSON object mapping standard field names to the actual column names:
{{
  "ticker": "Symbol",
  "name": "Description", 
  "shares": "Quantity",
  "price": "Last Price",
  "market_value": "Current Value"
}}

Only include fields you are confident about."""
        
        try:
            response = self.llm.complete(prompt, temperature=0.1)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Column mapping failed: {e}")
        
        return {}
    
    def _extract_date(self, text: str) -> Optional[date]:
        """Extract statement date from text."""
        patterns = [
            r'Statement Date[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'As of[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(date_str, fmt).date()
                        except ValueError:
                            continue
                except Exception:
                    continue
        
        return None
    
    def _detect_broker(self, text: str) -> Optional[str]:
        """Detect brokerage firm from text."""
        brokers = {
            'fidelity': 'Fidelity',
            'schwab': 'Charles Schwab',
            'vanguard': 'Vanguard',
            'interactive brokers': 'Interactive Brokers',
            'etrade': 'E*TRADE',
            'td ameritrade': 'TD Ameritrade',
            'robinhood': 'Robinhood',
            'webull': 'Webull'
        }
        
        text_lower = text.lower()
        for key, name in brokers.items():
            if key in text_lower:
                return name
        
        return None
    
    def _extract_cash_balance(self, text: str) -> Optional[float]:
        """Extract cash balance from text."""
        patterns = [
            r'Cash Balance[:]?\s*\$?([\d,]+\.?\d*)',
            r'Money Market[:]?\s*\$?([\d,]+\.?\d*)',
            r'Cash[:]?\s*\$?([\d,]+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except ValueError:
                    continue
        
        return None
    
    def _extract_account_value(self, text: str) -> Optional[float]:
        """Extract total account value from text."""
        patterns = [
            r'Account Value[:]?\s*\$?([\d,]+\.?\d*)',
            r'Total Value[:]?\s*\$?([\d,]+\.?\d*)',
            r'Portfolio Value[:]?\s*\$?([\d,]+\.?\d*)',
            r'Net Worth[:]?\s*\$?([\d,]+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except ValueError:
                    continue
        
        return None


# Singleton instance
document_parser = DocumentParser()
