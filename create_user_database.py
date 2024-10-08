 
import pickle
import hashlib
 
# Function to create a pickle database
def create_user_database():
    users = {
        'user1': hashlib.sha256('password1'.encode()).hexdigest(),
        'user2': hashlib.sha256('password2'.encode()).hexdigest()
    }
 
    with open('users.pkl', 'wb') as f:
        pickle.dump(users, f)
 
create_user_database()  # Run this once to create the database