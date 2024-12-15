import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import time
from langchain.schema import Document

# Your API key
API_KEY = '362ac3e5c8f3695a6f36c9af656a935af682b3bb52e1a0f914737358ca033d33'

# Sample data (you can add your own comments)
sample_data = [
    "You're so stupid, I can't stand you!",
    "You did an amazing job on the project!",
    "I'm going to make sure you regret this.",
    "Don't worry, you can do it! I believe in you.",
    "Stop being such a loser, no one cares about you.",
    "I think your presentation could be improved by adding more data visualizations."
]

# Convert the sample data into Document objects
documents = [Document(page_content=comment) for comment in sample_data]

# Split the sample data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Initialize Ollama embeddings model
embedding_model = OllamaEmbeddings()

# Embed the documents using the correct method
embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

# Create FAISS vector store
faiss_db = FAISS.from_texts([doc.page_content for doc in documents], embedding_model)

# Define a function to call the Meta-Llama API
def llama3_api_call(prompt):
    url = "https://api.together.xyz/v1/chat/completions"  # Correct API endpoint
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Correct model name
        "messages": [
            {
                "role": "system", 
                "content": """
                    You are a toxic comment detector and classifier. Your task is to assess whether the given comment is toxic or non-toxic and classify it into specific categories. 
                    If the comment is toxic, respond with the toxicity type and the reason, otherwise, respond with 'non-toxic' and the category of the comment. 
                    
                    Toxic categories:
                    1. Insults: Examples - Stupid, Idiot, Dumb, Moron, Fool, Worthless, Useless, Trash, Pathetic, Loser
                    2. Profanity: Examples - Damn, Hell, Crap, Bloody, Shit, Fuck, Asshole, Bastard, Bitch, Piss
                    3. Hate Speech: Examples - Racist, Bigot, Sexist, Homophobic, Xenophobic, Misogynist, Anti-Semitic, Discrimination, Intolerance, Prejudice
                    4. Threats: Examples - Kill, Attack, Harm, Hurt, Destroy, Die, Murder, Stab, Bomb, Explode
                    5. Harassment: Examples - Bully, Troll, Mock, Shame, Humiliate, Ridicule, Tease, Criticize, Abuse, Demean
                    6. Sexual Harassment: Examples - Creepy, Pervert, Obscene, Molest, Assault, Rape, Stalker, Lust, Voyeur, Predatory
                    7. Aggression: Examples - Angry, Rage, Violent, Fight, Smash, Punch, Shout, Scream, Argue, Break
                    8. Sarcasm/Passive Aggression: Examples - Sure, Right, Wow, Amazing, Great, Fine, Whatever, Really, Nice, Perfect
                    9. Derogatory Slurs: Examples - N-word, F-word, C-word, Slut, Whore, Retard, Fat, Ugly, Gay, Poor
                    10. Cyberbullying: Examples - Loser, Worthless, Dumb, Block, Ignore, Hate, Cancel, Expose, Shame, Report
                    
                    Non-toxic categories:
                    1. Compliments: Examples - Amazing, Beautiful, Brilliant, Excellent, Fantastic, Great, Lovely, Outstanding, Perfect, Wonderful
                    2. Encouragement: Examples - Keep going, Well done, You can do it, Don’t give up, Great job, Proud, Believe, Impressive, Fantastic work, Nice effort
                    3. Politeness: Examples - Please, Thank you, Sorry, Excuse me, Appreciate, Kind, Welcome, Respect, Grateful, Courteous
                    4. Empathy: Examples - Understand, Feel, Care, Support, Listen, Kindness, Compassion, Sympathy, Considerate, Warm
                    5. Friendliness: Examples - Hello, Hi, Nice to meet, Good day, Take care, Welcome, Greetings, Good evening, How are you, Pleasure
                    6. Constructive Feedback: Examples - Suggest, Recommend, Improve, Idea, Enhance, Adjust, Consider, Thought, Option, Suggestion
                    7. Neutral Questions: Examples - Why, How, What, When, Where, Can you, Could you, Would you, Is this, Does this
                    8. Positive Affirmations: Examples - Yes, Agree, Correct, True, Absolutely, Sure, Right, Certainly, Of course, Definitely
                    9. Motivational Phrases: Examples - You’ve got this, Don’t worry, It’s okay, Everything will be fine, Believe in yourself, Stay positive, Keep pushing, Keep smiling, Never give up, Bright future
                    10. Expressions of Gratitude: Examples - Thanks, Appreciate, Grateful, Thank you so much, Much obliged, Heartfelt thanks, Many thanks, Cheers, Blessed, Thankful
                """
            },
            {"role": "user", "content": prompt}  # User's input prompt
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', 'No response from model')
    else:
        return f"Error: {response.status_code}"

# Streamlit web interface
def main():
    st.set_page_config(page_title="Toxic Comment Detection Bot", page_icon=":question:", layout="centered")

    st.title("Toxic Comment Detection Bot")
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .css-1d391kg {
            text-align: center;
        }
        .stButton button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    with st.spinner('Loading...'):
        time.sleep(2)  # Simulating a delay for the animation

    st.success("You got your answer!")

    # Text input for user comment
    user_input = st.text_area("Enter your comment here:")

    # Button to trigger the toxic comment check
    if st.button('Check Toxicity'):
        if user_input:
            # Display the comment entered by the user
            st.write(f"User input: {user_input}")

            # Call the Meta-Llama API to check for toxicity
            response = llama3_api_call(user_input)
            
            # Display the result
            st.markdown(f"### {response}")
        else:
            st.warning("Please enter a comment to check.")

if __name__ == "__main__":
    main()
