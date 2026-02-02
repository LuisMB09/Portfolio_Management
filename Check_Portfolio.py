from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import threading
import time
import os
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

load_dotenv()


class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.account = None
        self.portfolio_rows = []
        self.account_values = []

    def managedAccounts(self, accountsList: str):
        self.account = accountsList.split(",")[0]
        self.reqAccountUpdates(True, self.account)

    def error(self, reqId, errorCode, errorString, *args):
        print("Error |", errorCode, errorString)

    def updateAccountValue(self, key, val, currency, accountName):
        self.account_values.append({
            "Account": accountName,
            "Key": key,
            "Value": val,
            "Currency": currency
        })

    def updatePortfolio(self, contract, position,
                        marketPrice, marketValue,
                        averageCost, unrealizedPNL,
                        realizedPNL, accountName):

        self.portfolio_rows.append({
            "Account": accountName,
            "Symbol": contract.symbol,
            "Currency": contract.currency,
            "Position": position,
            "MarketPrice": marketPrice,
            "MarketValue": marketValue,
            "AverageCost": averageCost,
            "UnrealizedPNL": unrealizedPNL,
            "RealizedPNL": realizedPNL
        })

    def accountDownloadEnd(self, accountName):
        portfolio_df = pd.DataFrame(self.portfolio_rows)
        account_df = pd.DataFrame(self.account_values)

        # ---- Portfolio analytics ----
        portfolio_df['Weight'] = portfolio_df['MarketValue'] / portfolio_df['MarketValue'].sum()
        portfolio_df['UnrealizedPNL_pct'] = (
            portfolio_df['MarketPrice'] / portfolio_df['AverageCost'] - 1
        )

        # ---- Print to console ----
        print("\n=== PORTFOLIO ===")
        print(portfolio_df, "\n")

        pnl = np.dot(
            portfolio_df['Weight'],
            portfolio_df['UnrealizedPNL_pct']
        )
        print(f"Portfolio Unrealized PnL: {pnl:.2%}\n")

        print("\n=== ACCOUNT VALUES ===")
        print(account_df, "\n")

        # ---- Create folder ----
        folder = "Portfolio_History"
        os.makedirs(folder, exist_ok=True)

        # ---- File name with date ----
        date_str = datetime.now().strftime("%d-%m-%Y")
        file_path = os.path.join(
            folder,
            f"Portfolio_Checkpoint_{date_str}.xlsx"
        )

        # ---- Write Excel ----
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            portfolio_df.to_excel(
                writer,
                sheet_name="Portfolio",
                index=False
            )
            account_df.to_excel(
                writer,
                sheet_name="Account_Values",
                index=False
            )

        print(f"📁 Portfolio snapshot saved to: {file_path}")

        self.stop()


    def stop(self):
        if self.account:
            self.reqAccountUpdates(False, self.account)
        self.disconnect()


def main():
    app = TestApp()
    app.connect(
        os.getenv("IBKR_HOST"),
        int(os.getenv("IBKR_PORT")),
        clientId=int(os.getenv("IBKR_CLIENT_ID"))
    )

    threading.Thread(target=app.run, daemon=True).start()
    time.sleep(10)


if __name__ == "__main__":
    main()
