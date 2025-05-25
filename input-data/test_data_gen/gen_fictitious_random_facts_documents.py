import json
import random

# Random data for generating fictitious documents
subjects = ["Foosal Computing", "Artificial Scalability", "Foozy Colonization", "Dance Travel", "Hyper Cloning"]
content_templates = [
    "{} is a revolutionary concept that may reshape the future of technology. It enables {} to {} in a way previously thought impossible.",
    "The development of {} has sparked debates among scientists. Many believe it could lead to {} within the next {} years.",
    "{} is currently being researched by leading institutes around the world. Early experiments suggest {} could {} .",
    "Some experts claim that {} could open doors to {}. However, there are still many challenges, including {}.",
    "{} could drastically alter our understanding of {}. With advancements in {}, this could become a reality."
]

# Generate a random fact about each subject
def generate_document(subject):
    template = random.choice(content_templates)
    fact = template.format(
        subject, 
        random.choice(["humanity", "machines", "societies", "ecosystems"]), 
        random.choice(["overcome barriers", "redefine physics", "explore new frontiers", "achieve unprecedented success"])
    )
    return fact

# Generate 5 random documents
documents = []
for i, subject in enumerate(subjects, start=1):
    content = generate_document(subject)
    documents.append({
        "id": i,
        "title": subject,
        "content": content
    })

    print(f"Document '{subject}': {content}")

"""
# Generate random questions for each document
def generate_questions(subject):
    return [
        f"What is {subject} and how does it impact future technology?",
        f"What challenges are associated with {subject}?"
    ]

questions = []
for i, subject in enumerate(subjects, start=1):
    questions.append({
        "doc_id": i,
        "questions": generate_questions(subject)
    })

# Save the documents to a JSON file
with open('fictitious_documents.json', 'w') as doc_file:
    json.dump(documents, doc_file, indent=4)

# Save the questions to a separate JSON file
with open('fictitious_questions.json', 'w') as q_file:
    json.dump(questions, q_file, indent=4)

print("Fictitious documents and questions generated and saved.")
"""