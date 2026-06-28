"""
NexaRetail – Business Sales Performance Analysis
================================================
Data Science & Analytics Internship Task – FY 2024
Author: [Your Name]
Tools: Python, Pandas, Matplotlib, Seaborn

This script:
  1. Generates a realistic sales dataset (5,000 transactions)
  2. Performs KPI analysis, trend analysis, and segmentation
  3. Produces 6 publication-quality charts
  4. Prints a business insight summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
from datetime import datetime, timedelta
import random
import os

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ── PALETTE ─────────────────────────────────────────────────────────────
BLUE   = '#2563eb'
GREEN  = '#059669'
VIOLET = '#7c3aed'
AMBER  = '#d97706'
RED    = '#dc2626'
CYAN   = '#0891b2'
PALETTE = [BLUE, VIOLET, GREEN, AMBER, RED, CYAN]
BG     = '#f7f8fc'
CARD   = '#ffffff'

# ── DATA GENERATION ──────────────────────────────────────────────────────
products = {
    "Technology":      {"Laptop Pro 15\"": (899,1299),"Wireless Headphones":(79,199),
                        "Mechanical Keyboard":(89,179),"USB-C Hub":(39,89),
                        "Webcam HD":(59,129),"Smart Speaker":(49,149),
                        "Monitor 27\"":(299,599),"Gaming Mouse":(45,99)},
    "Furniture":       {"Ergonomic Chair":(249,599),"Standing Desk":(399,899),
                        "Bookshelf 5-Tier":(89,179),"Filing Cabinet":(149,299),
                        "Desk Lamp":(39,89),"Monitor Stand":(49,129)},
    "Office Supplies": {"Notebook Pack":(12,29),"Ballpoint Pens (24pk)":(8,18),
                        "Sticky Notes Bulk":(15,35),"Stapler Pro":(19,49),
                        "Paper Shredder":(59,149),"Label Maker":(29,69),
                        "Whiteboard 4x3":(69,149),"Binder Set":(18,42)},
    "Clothing":        {"Business Shirt":(39,89),"Casual Sneakers":(59,139),
                        "Work Trousers":(49,109),"Wool Sweater":(69,149),
                        "Leather Belt":(25,65),"Formal Tie":(19,55)},
}
regions  = {"North":["Chicago","Detroit"],"South":["Houston","Atlanta"],
            "East":["New York","Boston"],"West":["LA","Seattle"]}
channels = ["Online","In-Store","Wholesale","Direct Sales"]
segments = ["Enterprise","SMB","Consumer","Government"]

def generate_data(n=5000):
    rows = []
    start = datetime(2024,1,1)
    for i in range(n):
        date = start + timedelta(days=random.randint(0,364))
        month = date.month
        seasonal = 1.6 if month in [11,12] else 1.2 if month in [6,7,8] else 0.8 if month in [1,2] else 1.0
        cat   = random.choices(list(products.keys()), weights=[35,20,30,15])[0]
        prod  = random.choice(list(products[cat].keys()))
        price = round(random.uniform(*products[cat][prod]),2)
        qty   = max(1,int(np.random.poisson(3)*seasonal))
        disc  = random.choices([0,.05,.10,.15,.20],[50,20,15,10,5])[0]
        rev   = round(price*qty*(1-disc),2)
        profit= round(rev*(1-random.uniform(.45,.70)),2)
        reg   = random.choice(list(regions.keys()))
        rows.append({"Order ID":f"ORD-{i+10000}","Date":date,
                     "Month":date.strftime("%b"),"Month_Num":date.month,
                     "Quarter":f"Q{(date.month-1)//3+1}","Category":cat,
                     "Product":prod,"Unit Price":price,"Quantity":qty,
                     "Discount":disc,"Revenue":rev,"Profit":profit,
                     "Region":reg,"City":random.choice(regions[reg]),
                     "Channel":random.choices(channels,[45,30,15,10])[0],
                     "Customer Segment":random.choices(segments,[25,35,30,10])[0],
                     "Customer ID":f"CUST-{random.randint(1000,9999)}"})
    return pd.DataFrame(rows)

print("Generating dataset …")
df = generate_data()
df.to_csv("sales_data.csv", index=False)
print(f"  ✓  {len(df):,} rows saved to sales_data.csv")

# ── KPI SUMMARY ───────────────────────────────────────────────────────────
total_rev    = df["Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = len(df)
avg_order    = df["Revenue"].mean()
margin       = total_profit / total_rev * 100
customers    = df["Customer ID"].nunique()

print("\n" + "═"*52)
print("  NEXARETAIL FY 2024 — EXECUTIVE SUMMARY")
print("═"*52)
print(f"  Total Revenue        ${total_rev:>12,.0f}")
print(f"  Total Profit         ${total_profit:>12,.0f}")
print(f"  Profit Margin        {margin:>11.1f}%")
print(f"  Total Orders         {total_orders:>12,}")
print(f"  Avg Order Value      ${avg_order:>12.2f}")
print(f"  Unique Customers     {customers:>12,}")
print("═"*52)

# ── AGGREGATIONS ──────────────────────────────────────────────────────────
monthly  = df.groupby(["Month_Num","Month"]).agg(Revenue=("Revenue","sum"),Profit=("Profit","sum"),Orders=("Order ID","count")).reset_index().sort_values("Month_Num")
cat_sum  = df.groupby("Category").agg(Revenue=("Revenue","sum"),Profit=("Profit","sum"),Orders=("Order ID","count")).reset_index().sort_values("Revenue",ascending=False)
reg_sum  = df.groupby("Region").agg(Revenue=("Revenue","sum"),Profit=("Profit","sum")).reset_index().sort_values("Revenue",ascending=False)
top10    = df.groupby(["Product","Category"]).agg(Revenue=("Revenue","sum"),Profit=("Profit","sum"),Qty=("Quantity","sum")).reset_index().sort_values("Revenue",ascending=False).head(10)
chan_sum  = df.groupby("Channel").agg(Revenue=("Revenue","sum"),Orders=("Order ID","count")).reset_index().sort_values("Revenue",ascending=False)

# ── PLOTTING ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22,18), facecolor=BG)
fig.suptitle("NexaRetail – Business Sales Performance Dashboard · FY 2024",
             fontsize=20, fontweight='bold', color='#0d1117', y=0.98)
gs = gridspec.GridSpec(3,3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.05, right=0.97, top=0.93, bottom=0.05)

def card_ax(subplot):
    ax = fig.add_subplot(subplot)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color('#e4e8f0')
    ax.tick_params(colors='#6b7280')
    return ax

def title(ax, t, s=""):
    ax.set_title(t, fontsize=13, fontweight='600', color='#0d1117', loc='left', pad=10)
    if s: ax.annotate(s, (0,1.02), xycoords='axes fraction', fontsize=10, color='#9ca3af')

# 1. Revenue & Profit Trend (wide)
ax1 = card_ax(gs[0,:2])
ax1.fill_between(monthly["Month"], monthly["Revenue"], alpha=0.12, color=BLUE)
ax1.plot(monthly["Month"], monthly["Revenue"], color=BLUE, lw=2.5, marker='o', ms=5, label='Revenue')
ax1.fill_between(monthly["Month"], monthly["Profit"], alpha=0.1, color=GREEN)
ax1.plot(monthly["Month"], monthly["Profit"], color=GREEN, lw=2, marker='s', ms=4, label='Profit')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
ax1.set_ylim(0)
ax1.grid(axis='y', color='#f3f4f6', linewidth=1)
ax1.legend(frameon=False, loc='upper left')
title(ax1,"Monthly Revenue & Profit Trend","Jan – Dec 2024")

# 2. Category Donut
ax2 = card_ax(gs[0,2])
wedges, texts, autotexts = ax2.pie(
    cat_sum["Revenue"], labels=cat_sum["Category"],
    colors=PALETTE[:4], autopct='%1.1f%%',
    pctdistance=0.82, startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2,"width":0.55}
)
for t in texts:    t.set_fontsize(10); t.set_color('#374151')
for t in autotexts: t.set_fontsize(9); t.set_fontweight('600')
ax2.add_artist(plt.Circle((0,0),0.4,color='white'))
title(ax2,"Revenue by Category")

# 3. Top 10 Products
ax3 = card_ax(gs[1,:2])
colors_bar = [BLUE if c=="Technology" else GREEN if c=="Furniture" else VIOLET if c=="Office Supplies" else AMBER for c in top10["Category"]]
bars = ax3.barh(top10["Product"][::-1], top10["Revenue"][::-1]/1000, color=colors_bar[::-1], height=0.65, edgecolor='none')
for bar in bars: ax3.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2, f'${bar.get_width():.0f}K', va='center', fontsize=9, color='#374151')
ax3.set_xlabel("Revenue ($K)", color='#6b7280', fontsize=10)
ax3.set_xlim(0, top10["Revenue"].max()/1000 * 1.18)
ax3.grid(axis='x', color='#f3f4f6')
ax3.tick_params(axis='y', labelsize=10)
title(ax3,"Top 10 Products by Revenue","Ranked by total FY 2024 revenue")

# 4. Region comparison
ax4 = card_ax(gs[1,2])
x = np.arange(len(reg_sum))
w = 0.38
ax4.bar(x-w/2, reg_sum["Revenue"]/1000, w, label='Revenue', color=BLUE, edgecolor='none', alpha=0.9)
ax4.bar(x+w/2, reg_sum["Profit"]/1000, w, label='Profit', color=GREEN, edgecolor='none', alpha=0.9)
ax4.set_xticks(x); ax4.set_xticklabels(reg_sum["Region"], fontsize=10)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
ax4.legend(frameon=False, fontsize=10); ax4.grid(axis='y', color='#f3f4f6')
title(ax4,"Revenue & Profit by Region")

# 5. Monthly Orders
ax5 = card_ax(gs[2,0])
ax5.bar(monthly["Month"], monthly["Orders"], color=VIOLET, alpha=0.85, edgecolor='none')
ax5.axhline(monthly["Orders"].mean(), color=RED, lw=1.5, ls='--', label=f'Avg {monthly["Orders"].mean():.0f}')
ax5.legend(frameon=False, fontsize=10); ax5.grid(axis='y', color='#f3f4f6')
ax5.tick_params(axis='x', rotation=45, labelsize=9)
title(ax5,"Monthly Order Volume")

# 6. Channel Revenue
ax6 = card_ax(gs[2,1])
ax6.bar(chan_sum["Channel"], chan_sum["Revenue"]/1000, color=PALETTE[:4], edgecolor='none', alpha=0.88)
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
ax6.grid(axis='y', color='#f3f4f6')
ax6.tick_params(axis='x', rotation=15, labelsize=9)
title(ax6,"Revenue by Sales Channel")

# 7. Quarterly
ax7 = card_ax(gs[2,2])
q = df.groupby("Quarter").agg(Revenue=("Revenue","sum"),Profit=("Profit","sum")).reindex(["Q1","Q2","Q3","Q4"])
x = np.arange(4); w = 0.38
ax7.bar(x-w/2, q["Revenue"]/1000, w, color=BLUE, alpha=0.9, edgecolor='none')
ax7.bar(x+w/2, q["Profit"]/1000, w, color=GREEN, alpha=0.9, edgecolor='none')
ax7.set_xticks(x); ax7.set_xticklabels(["Q1","Q2","Q3","Q4"])
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
ax7.grid(axis='y', color='#f3f4f6')
title(ax7,"Quarterly Performance")

plt.savefig("sales_analysis_charts.png", dpi=150, bbox_inches='tight', facecolor=BG)
print("\n  ✓  Charts saved to sales_analysis_charts.png")

# ── INSIGHTS REPORT ───────────────────────────────────────────────────────
print("\n" + "═"*52)
print("  KEY BUSINESS INSIGHTS")
print("═"*52)
top_cat  = cat_sum.iloc[0]
top_prod = top10.iloc[0]
top_reg  = reg_sum.iloc[0]
nov_rev  = df[df["Month_Num"]==11]["Revenue"].sum()
dec_rev  = df[df["Month_Num"]==12]["Revenue"].sum()
print(f"  1. {top_cat['Category']} is the #1 category: ${top_cat['Revenue']:,.0f}")
print(f"  2. {top_prod['Product']} is the top product: ${top_prod['Revenue']:,.0f}")
print(f"  3. {top_reg['Region']} leads regions: ${top_reg['Revenue']:,.0f}")
print(f"  4. Q4 (Nov+Dec) alone: ${nov_rev+dec_rev:,.0f} — holiday peak!")
print(f"  5. Online channel: ${df[df['Channel']=='Online']['Revenue'].sum():,.0f}")
print(f"  6. SMB segment: ${df[df['Customer Segment']=='SMB']['Revenue'].sum():,.0f}")
print("═"*52)
print("\n  Analysis complete. Check sales_analysis_charts.png\n")
