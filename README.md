# Adversarial Multimedia Recommendation System for Visual Recommnedation
![image](https://github.com/user-attachments/assets/7dc129b6-b79d-4398-a8ae-f962c9124c65)

# Problem Statement
In today’s digital economy, **visual recommendation systems** have become a cornerstone for industries where product appearance significantly influences customer decisions. Sectors such as **clothing and fashion, furniture and home décor, beauty and cosmetics, art and design, and food and beverage** heavily rely on visually-driven recommendations to engage customers and drive sales. Leading **e-commerce** platforms like **Amazon, Flipkart, and Alibaba, fashion retailers like Zara, H&M, and ASOS, and lifestyle platforms like Pinterest and Instagram Shopping** use advanced visual recommendation systems to provide personalized and visually appealing product suggestions.

These systems, powered by **deep neural networks (DNNs)**, analyze images and videos to learn rich visual representations that align product recommendations with user preferences. However, the dependence on deep learning models introduces a significant vulnerability: **adversarial attacks**.

By applying small, imperceptible changes (adversarial perturbations) to product images, malicious actors can manipulate these visual recommendation systems to:

- Promote counterfeit or low-quality products,
- Suppress genuine competitors' products,
- Mislead customers with irrelevant or deceptive product suggestions.
Such vulnerabilities threaten the accuracy, fairness, and trustworthiness of recommendation systems, ultimately damaging the user experience and the platform’s brand reputation.

To address this issue, the researchers propose a novel solution called **Adversarial Multimedia Recommendation (AMR)**. This method employs adversarial learning to enhance the system's robustness by training models to resist adversarial perturbations.

---
## How attackers influence user preference
![image](https://github.com/user-attachments/assets/5d05ed77-34e6-4743-bc30-0f91b1ad2ac3)

### Adversarial Attacks on Visual Recommendation Systems

### Example 1: Counterfeit Product Manipulation on Amazon

#### Scenario:
Amazon uses a visual recommendation system to suggest products based on images. A malicious seller wants their fake Gucci handbags to appear in the recommendations when customers browse authentic Gucci products.

#### How the Adversarial Attack Happens:
The seller uploads images of the counterfeit Gucci bags but adds tiny, imperceptible pixel changes to the product images. These tiny changes are invisible to the human eye but confuse Amazon’s image recognition system, making the fake bag seem visually similar to the authentic Gucci bags.<br>

As a result, when customers browse for real Gucci handbags, Amazon's recommender system mistakenly suggests the counterfeit bags alongside genuine products.

#### Impact of the Attack:
- Customers may unknowingly buy fake products, thinking they are authentic.
- Gucci’s brand reputation is harmed because customers might blame the brand for poor quality.
- Amazon’s trustworthiness declines because customers feel misled by its recommendations.

#### How AMR Could Prevent This:
If Amazon used the Adversarial Multimedia Recommendation (AMR) model, the system would be trained to detect and ignore these subtle manipulations, ensuring that only authentic Gucci bags appear in recommendations.

---

### Example 2: Boosting Sales of Low-Quality Products

#### Scenario:
A small electronics seller on Flipkart is selling low-quality Bluetooth headphones. These headphones aren't popular, so they rarely appear in product recommendations.

#### How the Adversarial Attack Happens:
The seller takes images of their headphones and adds adversarial noise (small, deliberate changes) to mimic the design features of popular brands like Sony or Bose.  <br>

Flipkart's visual recommendation system is tricked into thinking the cheap headphones are similar to high-end headphones.  
The seller’s headphones now appear in recommendations when users search for premium brands.

#### Impact of the Attack:
- Customers searching for Sony headphones might accidentally purchase the inferior product.
- High-end brands lose potential customers due to this unfair competition.
- The platform’s credibility drops because recommendations seem misleading.

#### How AMR Could Prevent This:
By using AMR, Flipkart’s system would be robust against these small manipulations. The fake similarity would be detected, and only relevant, high-quality recommendations would appear.

---

### Example 3: Seasonal Product Mismatch

#### Scenario:
A clothing seller on Myntra wants to sell unsold winter jackets during the summer season, where customers search for summer wear like T-shirts and shorts.

#### How the Adversarial Attack Happens:
The seller slightly alters the jacket images by adding tiny visual patterns similar to summer clothing (like bright colors or floral designs).  Myntra’s visual recommendation system is fooled into thinking the jackets resemble summer fashion.  The platform starts recommending winter jackets when customers search for summer clothing.

#### Impact of the Attack:
- Customers get annoyed seeing irrelevant winter clothes in summer searches.
- Sales conversion rates drop because recommendations are no longer aligned with the season.
- Myntra’s reputation suffers due to poor shopping experience.

#### How AMR Could Prevent This:
With AMR, the system would be trained to resist these minor manipulations, ensuring that seasonally appropriate products are always recommended.

---

### Example 4: Hiding Negative Reviews Through Image Manipulation

#### Scenario:
A seller on eBay is selling a product that has received many negative reviews due to poor quality. To avoid being flagged by the recommendation system, they try to trick the platform.

#### How the Adversarial Attack Happens:
The seller uploads a slightly altered version of the product image with invisible pixel changes.  eBay’s system treats this image as a new product, disconnecting it from past negative reviews.  The system begins recommending the product to more customers, despite its poor performance.

#### Impact of the Attack:
- More customers are exposed to a bad product without seeing honest feedback.
- Customers feel deceived and lose trust in eBay's recommendations.
- Genuine sellers are pushed out by those exploiting the system.

#### How AMR Could Prevent This:
By using AMR, eBay’s system would detect that the new image is almost identical to the poorly reviewed product, keeping the negative reviews linked and preventing the seller from hiding bad feedback.



