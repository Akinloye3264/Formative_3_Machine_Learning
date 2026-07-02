# 1. CSV PARSER  — read one record at a time, honouring quoted fields

def read_records(path):

    with open(path, encoding="utf-8") as f:
        content = f.read()

    fields = []          # completed fields for the current record
    field = ""           # characters of the current field
    in_quotes = False
    i = 0
    n = len(content)
    records = []

    while i < n:
        char = content[i]

        if in_quotes:
            if char == '"':
                # Is this an escaped quote ("") or the closing quote?
                if i + 1 < n and content[i + 1] == '"':
                    field += '"'
                    i += 2
                    continue
                in_quotes = False
                i += 1
            else:
                field += char
                i += 1
        else:
            if char == '"':
                in_quotes = True
                i += 1
            elif char == ",":
                fields.append(field)
                field = ""
                i += 1
            elif char == "\n":
                fields.append(field)
                records.append(fields)
                fields = []
                field = ""
                i += 1
            elif char == "\r":
                i += 1                       # ignore carriage returns
            else:
                field += char
                i += 1

    # flush the last field/record if the file didn't end with a newline
    if field or fields:
        fields.append(field)
        records.append(fields)

    return records



# 2. TOKENIZER  — a review's set of distinct words

def tokenize(text):

    text = text.lower().replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    words = set()
    current = ""
    for char in text.lower():
        if char.isalpha() or char == "'":
            current += char
        else:
            if current:
                words.add(current)
                current = ""
    if current:
        words.add(current)
    return words



# 3. LOAD + COUNT  — one pass, counts stored in plain dicts

def build_counts(path):
    records = read_records(path)

    # First row is the header: find which column is which.
    header = records[0]
    text_idx = header.index("review")
    label_idx = header.index("sentiment")

    n_total = 0
    n_positive = 0
    pos_word_counts = {}   # word -> # positive reviews containing it
    all_word_counts = {}   # word -> # reviews (any label) containing it

    for row in records[1:]:
        if len(row) <= max(text_idx, label_idx):
            continue                          # skip any malformed row
        label = row[label_idx].strip().lower()
        words = tokenize(row[text_idx])

        n_total += 1
        for word in words:
            all_word_counts[word] = all_word_counts.get(word, 0) + 1

        if label == "positive":
            n_positive += 1
            for word in words:
                pos_word_counts[word] = pos_word_counts.get(word, 0) + 1

    return n_total, n_positive, pos_word_counts, all_word_counts



# 4. BAYES' THEOREM

def bayes_positive_given_keyword(keyword, n_total, n_positive, pos_word_counts, all_word_counts):
    keyword = keyword.lower()

    prior = n_positive / n_total                      # P(Positive)
    likelihood = pos_word_counts.get(keyword, 0) / n_positive   # P(kw|Pos)
    marginal = all_word_counts.get(keyword, 0) / n_total        # P(kw)

    if marginal == 0:
        posterior = 0.0
    else:
        posterior = (likelihood * prior) / marginal   # P(Pos|kw)

    return prior, likelihood, marginal, posterior



# 5. RUN

def main():
    positive_keywords = ["brilliant", "masterpiece", "wonderful", "excellent"]
    negative_keywords = ["boring", "terrible", "waste", "awful"]

    n_total, n_positive, pos_counts, all_counts = build_counts(
        "bayesian\IMDB_Dataset.csv")

    print("Loaded", n_total, "reviews (", n_positive, "positive )\n")

    header = ("Keyword".ljust(12) + "Prior".rjust(9) + "Likelihood".rjust(12)
              + "Marginal".rjust(10) + "Posterior".rjust(11))
    print(header)
    print("-" * len(header))

    for keyword in positive_keywords + negative_keywords:
        prior, likelihood, marginal, posterior = bayes_positive_given_keyword(
            keyword, n_total, n_positive, pos_counts, all_counts)
        print(keyword.ljust(12)
              + ("%.4f" % prior).rjust(9)
              + ("%.4f" % likelihood).rjust(12)
              + ("%.4f" % marginal).rjust(10)
              + ("%.4f" % posterior).rjust(11))


if __name__ == "__main__":
    main()