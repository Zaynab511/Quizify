import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import requests
import pickle
import hashlib
import login  # Import the login module
test_text = """
welcome back to another machine learning explained video by assembly ai in this video we talk about supervised learning which is arguably the most important type of machine learning you will learn what it means examples of supervised learning or this data and training types of supervised learning and we touch on specific algorithms of supervised learning let's begin with the very basics what does machine learning mean machine learning is a sub-area of artificial intelligence and it's the study of algorithms that give computers the ability to learn and make decisions based on data and not from explicit instructions a popular example is learning to predict whether an email is spam or no spam by reading many different emails of these two types we typically differentiate between three types of machine learning supervised learning unsupervised learning and reinforcement learning in supervised learning the computer learns by making use of labeled data so we know the corresponding label or target of our data an example is again the spam prediction algorithm where we show many different emails to the computer and for each email we know if this was a spam email or not on the other hand in unsupervised learning the computer learns by making use of unlabeled data so we have data but we don't know the corresponding target an example is to cluster books into different categories on the basis of the title and other book information but not by knowing its actual category and then there is also reinforcement learning where so-called intelligent software agents take actions in an environment and automatically try to improve its behavior this usually works with a system of rewards and punishments and popular examples are games for example a computer can learn to be good in the snake game only by playing the game itself and every time
it eats an apple or it dies it learns from this actions now in this video we are going to focus on supervised learning where we learn from labeled data now what is data data can be any relevant information we collect for our algorithm this can be for example user information like age and gender or text data or images or information within an image like measurements or color information the possibilities are endless here let's look at a concrete example in the popular iris flower data set we want to predict the type of iris flower based on different measurements we have 150 records of flowers with different attributes that have been measured before so for each flower we have the sepal
length saypal width petal length and petal width these are called the features and we also have the corresponding species this is called the class the label or the target so this is a supervised case where we know the label we can
represent this table in a mathematical way so we put each feature into a vector this is the feature vector and then we do this for all the different samples and when we do this for all the different samples we end up in a 2d representation which is also called a matrix additionally we can put all labels into one vector this is called the target vector now in supervised learning we take the features and the labels and show it to the computer so that it learns we call this the training step and the data we use is called the training data training is performed by specific algorithms that usually try to minimize an error during this training process and this is done by mathematical optimization methods which i won't go into more detail here after training we want to show new data to the computer that it has never seen before and where we don't know the label this is called our test data and now the trained computer should be able to make a decision based on the information it has seen and determine the correct target value and this is how supervised learning works there are two types of supervised learning classification and regression in classification we predict a discrete class label in the previous flower classification example our target values can only have the values 0 1 and 2 corresponding to the three different classes if we have more than two possible labels like here we call this a multi-class classification problem if we only have two labels usually zero and one is used then we call this a binary classification problem for example spam or no spam on the other hand in regression we try to predict a continuous target value meaning the target value can have a more or less arbitrary value one example is to predict house prices based on given information about the house and the neighborhood the target variable which is the price can basically have any value here now that we know what supervised learning means let's have a look at concrete algorithms i will not explain them in detail here i simply name them so that you have heard of them they all have a unique design and can be different in the way how it stores the information mathematically how it solves the training process through mathematical operations and how it transforms the data this list is not exhaustive but here are 10 algorithms that are nice to know some of them can be used for either regression or classification and some can even be used for both cases popular algorithms are linear regression logistic regression decision trees random forest naive bayes perceptron and multi-layer perceptron support vector machines or short svm k-nearest neighbors or short knn adaboost and neural networks which are part of the deep learning field alright i hope you enjoyed
this video if you did so then please hit the like button and consider subscribing to the channel also if you want to try assembly ai for free then grab your free api token using the link in the description below and then i hope to
see you in the next video bye
"""
users = login.load_users()
# Initialize session state variable for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Check if user is authenticated
if not st.session_state.authenticated:
    # Show login page
    login.login(users)
else:
    # Your existing app code below
    load_dotenv()
 
    api_key = os.environ.get("API_KEY")
    endpoint = os.environ.get("END_POINT")
    QUIZ_PICKLE_FILE = "quiz_data.pkl"
    USER_RESULTS_FILE = "user_results.pkl"
    ATTEMPT_TRACKER_FILE = "attempt_data.pkl"
    USER_DATA_FILE = "user_data.pkl"
    # Function to save MCQs to a pickle file
    def save_quiz_to_pickle(mcqs, filename):
        with open(filename, 'wb') as f:
            pickle.dump(mcqs, f)

    # Function to load MCQs from a pickle file
    def load_quiz_from_pickle(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return []

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
                # Define the MCQ and List_of_MCQs data models
    class MCQ(BaseModel):
        question: str
        options: List[str]
        answer: str

    class List_of_MCQs(BaseModel):
        mcqs: List[MCQ]
        
    def split_text(text, max_length):
        """Split text into chunks of a maximum length."""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]
    def generate_mcqs_from_text(text):
        """Generate MCQs from text using OpenAI."""
        example_json = [
            {
                "question": "What is the capital of France?",
                "options": ["Paris", "London", "Berlin", "Madrid"],
                "answer": "Paris"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "options": ["Earth", "Mars", "Jupiter", "Saturn"],
                "answer": "Mars"
            }
        ]

        text_chunks = split_text(text, 1000)
        all_mcqs = []
        for chunk in text_chunks:
            prompt = f"Generate multiple-choice questions (MCQs) from the following text in JSON format:\n\n{chunk}"
            messages = [
                {"role": "system", "content": f"Generate 2 MCQs from the following text in JSON format:\n{json.dumps(example_json, indent=4)}"},
                {"role": "user", "content": chunk}
            ]
            response = get_chat_completion(messages)
            try:
                mcqs = json.loads(response)
                if 'MCQs' in mcqs:
                    all_mcqs.extend(mcqs['MCQs'])
                elif 'questions' in mcqs:
                    all_mcqs.extend(mcqs['questions'])
            except KeyError as e:
                print(f"KeyError: {str(e)} - response did not have expected key.")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {str(e)} - response could not be parsed.")
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
            finally:
                continue
        
        return all_mcqs
        
    # Improved get_chat_completion
    def get_chat_completion(messages):
        data = {
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', None)
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None

    # Function to load attempt data from pickle
    def load_attempt_data(file):
        if os.path.exists(file):
            with open(file, "rb") as f:
                return pickle.load(f)
        return {}

    # Function to save attempt data to pickle
    def save_attempt_data(data, file):
        with open(file, "wb") as f:
            pickle.dump(data, f)

        # Function to load user data (for quiz attempts) from pickle
# Function to load user data (for quiz attempts) from pickle
    def load_user_data(file):
        if os.path.exists(file) and os.path.getsize(file) > 0:  # Check if file exists and is not empty
            with open(file, "rb") as f:
                return pickle.load(f)
        return {}  # Return an empty dictionary if the file does not exist or is empty

    # Function to save user data (for quiz attempts) to pickle
    def save_user_data(data, file):
        with open(file, "wb") as f:
            pickle.dump(data, f)
    attempt_data = load_attempt_data(ATTEMPT_TRACKER_FILE)

    # Initialize session state
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None
    if "attempted_quizzes" not in st.session_state:
        st.session_state.attempted_quizzes = []
    # Initialize session state variables if not already done
    if "all_mcqs" not in st.session_state:
        st.session_state.all_mcqs = []
    if "username" not in st.session_state:
        st.session_state.username = None  # Assuming a username is set during login
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = {}
    def load_quizzes(filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return []
    # Function to load user quiz results from the pickle file
    def load_user_results():
        try:
            with open('user_data.pkl', 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}  # Return an empty dictionary if the file does not exist or is empty

    # Function to save user quiz results to the pickle file
    def save_user_results(username, quiz_title, score):
        try:
            user_results = load_user_results()  # Load existing results
            # Append new result for the user
            if username not in user_results:
                user_results[username] = []
            # Only add the score if the quiz hasn't been attempted
            if quiz_title not in [result[0] for result in user_results[username]]:
                user_results[username].append((quiz_title, score))

            with open('user_data.pkl', 'wb') as f:
                pickle.dump(user_results, f)  # Save updated results
        except Exception as e:
            print(f"Error saving user results: {e}")

    # Function to get all quiz scores for highest score display
    def get_highest_scores():
        user_results = load_user_results()
        highest_scores = {}
        
        for user, results in user_results.items():
            for quiz_title, score in results:
                if quiz_title not in highest_scores:
                    highest_scores[quiz_title] = score
                else:
                    highest_scores[quiz_title] = max(highest_scores[quiz_title], score)
        
        return highest_scores
    # Load all MCQs from the pickle file
    st.session_state.all_mcqs = load_quiz_from_pickle(QUIZ_PICKLE_FILE)
    # Function to calculate the checksum of a file or string
    def calculate_checksum(file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    def calculate_checksum_from_string(input_string):
        return hashlib.md5(input_string.encode('utf-8')).hexdigest()

    st.title("PDF and YouTube Quiz Generator")

    tab1, tab3, tab4 = st.tabs(["PDF to MCQs", "Quiz Results", "User Profile"])

    with tab1:
        st.header("PDF to MCQs")

        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if 'score' not in st.session_state:
            st.session_state.score = 0  # Initialize score

        # Initialize the MCQs in session state if it doesn't exist
        if "all_mcqs" not in st.session_state:
            st.session_state.all_mcqs = []

        if st.session_state.current_text == "" and not st.session_state.all_mcqs:
            if uploaded_file is not None:
                with open("uploaded_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_reader = PyPDFLoader("uploaded_file.pdf")
                documents = pdf_reader.load()
                pdf_text = "\n".join([doc.page_content for doc in documents])
                os.remove("uploaded_file.pdf")
                st.session_state.current_text = pdf_text
                st.session_state.all_mcqs = generate_mcqs_from_text(pdf_text)
                st.rerun()
        else:
            st.session_state.all_mcqs = []  # Reset if no file uploaded

        if uploaded_file is not None:
            quiz_title = uploaded_file.name.replace(".pdf", "")
            st.subheader(f"Quiz Title: {quiz_title}")
            file_bytes = uploaded_file.getvalue()
            new_checksum = calculate_checksum(file_bytes)
            
            # Check if MCQs for this PDF already exist
            existing_mcqs = [mcq for mcq in st.session_state.all_mcqs if mcq.get('checksum') == new_checksum]
            if existing_mcqs:
                st.warning("MCQs for this PDF already exist.")
                st.session_state.all_mcqs = existing_mcqs
            else:
                with open("uploaded_file.pdf", "wb") as f:
                    f.write(file_bytes)

                pdf_reader = PyPDFLoader("uploaded_file.pdf")
                documents = pdf_reader.load()
                pdf_text = "\n".join([doc.page_content for doc in documents])
                os.remove("uploaded_file.pdf")
                
                new_mcqs = generate_mcqs_from_text(pdf_text)
                for mcq in new_mcqs:
                    mcq['checksum'] = new_checksum

                st.session_state.all_mcqs.extend(new_mcqs)
                save_quiz_to_pickle(st.session_state.all_mcqs, QUIZ_PICKLE_FILE)
                st.success("Generated and saved MCQs.")


            if st.session_state.all_mcqs:
                all_mcqs = st.session_state.all_mcqs
                current_question = st.session_state.current_question
                quiz_title = uploaded_file.name.replace(".pdf", "") if uploaded_file else "Current Quiz"

                if quiz_title not in attempt_data:
                    attempt_data[quiz_title] = []

                if current_question < len(all_mcqs):
                    mcq = all_mcqs[current_question]
                    st.write(f"**Question {current_question + 1}:** {mcq['question']}")
                    selected_option = st.radio("Select an option:", mcq['options'], key=f"question_{current_question}")

                    if st.button("Submit"):
                        st.session_state.selected_option = selected_option
                        st.session_state.show_feedback = True

                    if st.session_state.show_feedback:
                        if st.session_state.selected_option == mcq['answer']:
                            st.success("Correct!")
                            st.session_state.score += 1
                        else:
                            st.error(f"Wrong! The correct answer is: {mcq['answer']}")

                        if st.button("Next"):
                            st.session_state.current_question += 1
                            st.session_state.show_feedback = False
                            st.session_state.selected_option = None
                            st.rerun()
                else:
                    # Quiz completed
                    st.write(f"Quiz completed! Your score is {st.session_state.score} out of {len(all_mcqs)}.")

                    # Record the attempt
                    attempt_data[quiz_title].append({
                        'attempt_number': len(attempt_data[quiz_title]) + 1,
                        'score': st.session_state.score,
                        'total_questions': len(all_mcqs)
                    })
                    save_attempt_data(attempt_data, ATTEMPT_TRACKER_FILE)

                # Ensure username is defined before using it
                username = st.session_state.get('username')

                if username:  # Proceed only if username is defined
                    user_profile_data = load_user_data(USER_DATA_FILE)
                    if username not in user_profile_data:
                        user_profile_data[username] = {'name': username, 'quizzes': []}

                    user_profile_data[username]['quizzes'].append({
                        'title': quiz_title,
                        'score': st.session_state.score,
                        'total_questions': len(all_mcqs)
                    })

                    save_user_data(user_profile_data, USER_DATA_FILE)
                else:
                    st.warning("Please log in to save your quiz results.")


                    # Show the reattempt button
                    if st.button("Reattempt Quiz"):
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.selected_option = None
                        st.session_state.show_feedback = False
                        st.rerun()

                    # Display attempt history
                    st.write("### Attempt History")
                    for attempt in attempt_data[quiz_title]:
                        st.write(f"Attempt {attempt['attempt_number']}: Score: {attempt['score']}/{attempt['total_questions']}")

            else:
                st.write("Please upload a PDF file to generate MCQs.")

    with tab3:
        st.header("Quiz Results")
        user_results = load_user_results()
        highest_scores = get_highest_scores()

        all_quizzes = load_quizzes(QUIZ_PICKLE_FILE)
        quiz_titles = set(quiz.get('title') for quiz in all_quizzes if 'title' in quiz)


        for quiz_title in quiz_titles:
            st.subheader(f"Results for '{quiz_title}':")
            scores = [(user, score) for user, results in user_results.items() for title, score in results if title == quiz_title]
            if scores:
                highest_score = highest_scores.get(quiz_title, 0)
                st.write(f"Highest Score: {highest_score}")
                for user, score in scores:
                    st.write(f"User: {user} - Score: {score}")
                if st.button(f"Attempt '{quiz_title}' Again"):
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.all_mcqs = next(quiz['questions'] for quiz in all_quizzes if quiz['title'] == quiz_title)
                    st.session_state.quiz_title = quiz_title
                    st.experimental_rerun()
            else:
                st.write("No attempts yet.")

    with tab4:
        st.header("User Profile")
        username = st.session_state.get('username')  # Assuming you store the username after login
        st.write(f"Username: {username}")
        # Display additional user profile information here if needed

    st.write("---")