# 🌊 **TravelTide: Personalized Perk Program for User Engagement**

---

## 📄 **Context**  
TravelTide is an innovative travel platform looking to increase user engagement and encourage bookings through personalized reward mechanisms. With rich user-level and session-level behavioral data, the company aims to create a perk-based incentive system tailored to individual user profiles and motivations.

---

## 🎯 **Objective**  
The main objective was to design a data-driven perk assignment strategy that maximizes user engagement while aligning with TravelTide’s business goals. This includes developing:

- A clustering model to segment users into meaningful behavioral groups  
- Personalized perk allocation based on predictive scores and eligibility  
- A Tableau dashboard for stakeholder-facing storytelling and decision-making  

---

## 🚀 **Approach and Implementation**  

### 1. **Data Preparation (SQL & Python)**  
- Extracted and transformed data from four main tables: `users`, `flights`, `hotels`, and `sessions`  
- Applied normalization, feature engineering, and derived metrics like travel duration, booking lead time, and price categories  
- Designed two main datasets:
  - `session_level_based_table_cleaned.csv`
  - `user_level_based_table_clustered_final.csv`

### 2. **Clustering & Scoring (Python)**  
- Created six distinct clusters using KMeans on perk-related features (scores + eligibility)  
- Defined custom perks for each cluster based on travel behavior and engagement patterns  
- Assigned perks conditionally using score thresholds, with lower barriers for key strategic segments  

### 3. **Perk Distribution & User Table**  
- Final user-level table includes:
  - Demographics and loyalty indicators  
  - Booking behavior (e.g., total trips, costs, durations)  
  - Perk scores and eligibility flags  
  - Cluster ID, assigned perk, and marketing-friendly segment name  

### 4. **Tableau Dashboard**  
- Multi-sheet Tableau dashboard visualizing:
  - Cluster performance and characteristics  
  - Perk distribution and eligibility  
  - Segment-specific KPIs  
  - Executive overview and strategic recommendations  

![Tableau Dashboard Mockup](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/Tableau_MockUp.png)

---

## 🔍 **Insights**  

- **Cluster-Based Targeting:** Six behavioral user segments with unique motivations  
- **Smart Perk Assignment:** Targeted perks (e.g., free bags, discounts, drinks) are offered based on behavioral scoring  
- **Booking Patterns:** Long-term loyal users prefer flexibility and bundled perks  
- **Strategic Segments:** Segments like *The Frequent Flyers* and *The Savvy Planners* offer high conversion potential  

---

## 📊 **Project Steps**

1. **SQL Preprocessing**  
   - Transformed raw PostgreSQL tables into feature-rich analytical tables  
   - Mapped fare and hotel price categories, page click activity, booking intent  

2. **Python-Based Modeling**  
   - Designed perk scores for 6 reward types  
   - Applied KMeans clustering (6 segments), validated with silhouette score and DBI  
   - Assigned perks using strategic thresholds tailored to each segment  

3. **Data Visualization**  
   - Interactive Tableau dashboard presenting user segmentation and perk logic  
   - Each segment has a dedicated sheet with filters, metrics, and descriptions  

---

## 🧠 **Standout Logic**  
- **Segment-Specific Thresholds:** Perks are only granted to users who exceed cluster-specific behavioral thresholds  
- **No Threshold for Key Groups:** Segments like *No-Bookers* and *Low-Activity Users* receive perks with no threshold to stimulate engagement  
- **Marketing-Friendly Naming:** Cluster names such as *The Curious Visitors* and *The Premium Travelers* improve storytelling and presentation impact  

---

## 📋 **Executive Summary**  
- [One-Page Executive Summary PDF](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/Executive_Summary.pdf)

---

## 📋 **Detailed Report**
- [Three-Page Detailed Report PDF](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/Detailed%20Report.pdf)
---

## 🔗 **Links**

- **🔗 Tableau Dashboard:** [View Dashboard](https://public.tableau.com/views/TravelTide_Mastery-Project/TravelTide_PerkRewardProgram?:language=de-DE&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- **📂 Cleaned Session-Level Table:** [`session_level_based_table_cleaned.csv`](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/session_level_based_table_cleaned.csv)  
- **📂 Final User Table (with Clusters):** [`user_level_based_table_clustered_final.csv`](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/user_level_based_table_clustered_final.csv) 
- **📄 SQL Queries:** [`final_queries_full.sql`](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/final_queries_full.sql)
- **📄 Python Code:** [`perk_assignment_from.py`](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/perk_assignment_from.py)  
- **📔 Colab Notebook:** [`TravelTide_Mastery-Project_Sebastian-Bangemann.ipynb`](https://colab.research.google.com/drive/1d7Wfw7gJsM385mHAcEsV1DRHhq4hRv3I?usp=sharing)
- **🎬 Presentation Video:** [Watch Video](https://drive.google.com/file/d/1_TGIfibhuzn9QmX6iy6Go2_dNZaCuZ55/view?usp=drive_link)
- **🧾 Presentation Slides:** [Open Slides](https://drive.google.com/file/d/1N68kFPBd57dgD29F2VlMyUgUgf_n_14H/view?usp=drive_link)

---

## ✅ **Recommendations**

- **Launch Perk Campaign:** Reach out to eligible users with personalized perks  
- **Monitor Performance:** Track how perks impact engagement and booking behavior over the next 6 months  
- **Run Awareness Campaigns:** Highlight the perk program to other customers to build interest  
- **Expand Over Time:** Consider seasonal perks, loyalty tiers, and broader user segments in future phases  

---

💡 **Thank you for reviewing the TravelTide perk segmentation project!**  
Let’s turn data into action — one perk at a time.

See [DISCLAIMER.md](https://github.com/seb-bange/travel-tide_mastery-project/blob/main/DISCLAIMER) for usage considerations.
