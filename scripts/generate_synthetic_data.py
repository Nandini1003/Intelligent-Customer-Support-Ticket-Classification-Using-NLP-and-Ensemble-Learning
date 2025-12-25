import random
import pandas as pd

# Categories and templates
categories = {
    "Billing": ["I was charged twice", "Invoice not received", "Payment failed"],
    "Technical Issue": ["Cannot login to account", "System error on page", "App crashes frequently"],
    "Refund": ["I want a refund", "Request reimbursement", "Money back not processed"],
    "Account": ["Change my account details", "Update my profile info", "Unable to reset password"],
    "Delivery / Service": ["Delivery delayed", "Package not received", "Service not satisfactory"],
    "General Inquiry": ["Need information about product", "Question about services", "Operating hours?"]
}

# Synonyms to replace words
synonyms = {
    "refund": ["reimbursement", "money back"],
    "billing": ["payment", "invoice"],
    "technical": ["system", "login", "error", "issue"],
    "account": ["profile", "user account", "login"],
    "delivery": ["shipment", "package", "order"],
    "service": ["support", "assistance"],
    "general": ["information", "query", "question"]
}

def add_typos(text, typo_prob=0.1):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < typo_prob and len(word) > 3:
            idx = random.randint(0, len(word)-2)
            # Swap two letters
            word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
        new_words.append(word)
    return " ".join(new_words)

def replace_synonyms(text, synonym_prob=0.3):
    words = text.split()
    new_words = [random.choice(synonyms.get(w, [w])) if random.random() < synonym_prob else w for w in words]
    return " ".join(new_words)

def shuffle_words(text, shuffle_prob=0.2):
    words = text.split()
    if random.random() < shuffle_prob:
        random.shuffle(words)
    return " ".join(words)

def generate_ticket(category):
    base_text = random.choice(categories[category])
    
    # Apply noise
    text = add_typos(base_text)
    text = replace_synonyms(text)
    text = shuffle_words(text)
    
    return text

def generate_dataset(num_samples_per_category=600):
    data = []
    for cat in categories.keys():
        for _ in range(num_samples_per_category):
            # Randomly decide if this ticket is mixed intent (20% chance)
            if random.random() < 0.2:
                # Pick a second random category
                other_cat = random.choice([c for c in categories.keys() if c != cat])
                text = generate_ticket(cat) + " and " + generate_ticket(other_cat)
            else:
                text = generate_ticket(cat)
            
            data.append({"customer_message": text, "category": cat})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/raw/synthetic_support_tickets_noisy.csv", index=False)
    print("Noisy synthetic dataset generated:", df.shape)
