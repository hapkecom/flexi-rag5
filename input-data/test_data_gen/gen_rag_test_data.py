from factory.llm_factory import get_default_embeddings, get_default_chat_llm_without_streaming
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from datasets import Dataset 
import logging

logger = logging.getLogger(__name__)


def generate_test_data():
    # preparation
    test_size = 10
    generator_llm = get_default_chat_llm_without_streaming()
    critic_llm = get_default_chat_llm_without_streaming()
    ollama_emb = get_default_embeddings()
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=ollama_emb
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(documents, test_size=test_size, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, raise_exceptions=False)


    # Once you have a testset created, you can load it as a Dataframe, analyse it & save it in a csv to use later for evaluation.
    test_df = testset.to_pandas()
    test_df.head()

def generate_test_data_from_example():
    data_samples = {
        'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
        'answer': ['The first superbowl was held on January 15, 1967', 'The most super bowls have been won by The New England Patriots'],
        'contexts' : [['The Super Bowl....season since 1966,','replacing the NFL...in February.'], 
        ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
        'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
    }
    dataset = Dataset.from_dict(data_samples)
    logger.info(f"dataset={dataset}")



def v2():
    import random

    # Mock data for ragas
    ragas = [
        {"name": "Bhairav", "time": "Morning", "mood": "serene", "thaat": "Bhairav", "notes": "Sa Re Ga Ma Pa Dha Ni"},
        {"name": "Yaman", "time": "Evening", "mood": "devotional", "thaat": "Kalyan", "notes": "Ni Re Ga Ma Pa Dha Ni"},
        {"name": "Bageshree", "time": "Late Night", "mood": "romantic", "thaat": "Kafi", "notes": "Sa Ga Ma Pa Dha Ni"},
        {"name": "Durga", "time": "Night", "mood": "joyful", "thaat": "Bilawal", "notes": "Sa Re Ma Pa Dha Sa"},
        {"name": "Marwa", "time": "Dusk", "mood": "serious", "thaat": "Marwa", "notes": "Re Ga Ma Pa Dha Ni"},
        {"name": "Kafi", "time": "Any time", "mood": "melancholic", "thaat": "Kafi", "notes": "Sa Re Ga Ma Pa Dha Ni"},
        {"name": "Desh", "time": "Monsoon", "mood": "joyful", "thaat": "Khamaj", "notes": "Sa Re Ma Pa Dha Ni Sa"},
        {"name": "Hamsadhwani", "time": "Evening", "mood": "auspicious", "thaat": "Bilawal", "notes": "Sa Re Ga Pa Ni Sa"},
        {"name": "Malkauns", "time": "Midnight", "mood": "serious", "thaat": "Bhairavi", "notes": "Sa Ga Ma Dha Ni Sa"},
        {"name": "Todi", "time": "Morning", "mood": "serious", "thaat": "Todi", "notes": "Re Ga Ma Dha Ni Sa"}
    ]

    # Mock questions to generate test cases
    questions = [
        "What is the mood of {raga} raga?",
        "Which time of day is suitable for performing {raga}?",
        "Which notes are used in {raga}?",
        "What is the thaat of {raga}?",
        "Describe the characteristic of {raga}.",
    ]

    # A simple function to simulate the RAG system response generation
    def rag_system_response(question, raga_info):
        if "mood" in question:
            return f"The mood of {raga_info['name']} is {raga_info['mood']}."
        elif "time" in question:
            return f"{raga_info['name']} is performed during {raga_info['time']}."
        elif "notes" in question:
            return f"The notes of {raga_info['name']} are {raga_info['notes']}."
        elif "thaat" in question:
            return f"The thaat of {raga_info['name']} is {raga_info['thaat']}."
        elif "characteristic" in question:
            return f"{raga_info['name']} is from the {raga_info['thaat']} thaat, performed at {raga_info['time']}, with a {raga_info['mood']} mood."
        return "Unknown question."

    # Generate test cases
    def generate_test_cases(num_cases=10):
        test_cases = []
        
        for _ in range(num_cases):
            # Randomly select a raga and a question template
            raga_info = random.choice(ragas)
            question_template = random.choice(questions)
            
            # Format the question
            question = question_template.format(raga=raga_info["name"])
            
            # Generate the expected response from the RAG system
            expected_response = rag_system_response(question, raga_info)
            
            # Create a test case
            test_case = {
                "question": question,
                "expected_response": expected_response,
                "raga_info": raga_info
            }
            
            # Append to test cases
            test_cases.append(test_case)
        
        return test_cases

    # Print test cases
    test_cases = generate_test_cases()
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Question: {test_case['question']}")
        print(f"Expected Response: {test_case['expected_response']}")
        print("-" * 50)


generate_test_data_from_example()
