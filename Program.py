import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, ttk

# Function bach nchargiw data
def load_data(filepath):
    data = pd.read_csv(filepath, names=['userId', 'productId', 'Rating', 'timestamp'])
    data.drop('timestamp', axis=1, inplace=True)
    return data

# Function bach nfiltriw data 3la asass top N users w top M products
def filter_data(data, num_users=10000, num_products=10000):
    top_users = data['userId'].value_counts().head(num_users).index
    filtered_data = data[data['userId'].isin(top_users)]
    top_products = filtered_data['productId'].value_counts().head(num_products).index
    filtered_data = filtered_data[filtered_data['productId'].isin(top_products)]
    return filtered_data

# Function bach nkawnu user-item matrix
def create_user_item_matrix(filtered_data):
    user_item_matrix = filtered_data.pivot(index='userId', columns='productId', values='Rating')
    user_item_matrix.fillna(0, inplace=True)
    return user_item_matrix

# Function bach nhasbou item similarity matrix
def compute_item_similarity(user_item_matrix):
    sparse_user_item_matrix = csr_matrix(user_item_matrix.values)
    item_similarity = cosine_similarity(sparse_user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

# Function bach ndirou item-based recommendations
def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, n=10):
    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings[user_ratings > 0]
    scores = user_ratings.dot(item_similarity_df.loc[user_ratings.index, :])
    scores = scores.sort_values(ascending=False)
    recommended_items = scores.index.difference(user_ratings.index)
    return scores.loc[recommended_items].head(n)

# Nchargiw w n3aljew data
electronics_data = load_data("ratings_Electronics.csv")
filtered_data = filter_data(electronics_data)
user_item_matrix = create_user_item_matrix(filtered_data)
item_similarity_df = compute_item_similarity(user_item_matrix)

# Nt'akwdo mn tatalom indices
user_item_matrix = user_item_matrix.loc[:, item_similarity_df.index]

# Function bach n3amrou l button w n'affichiw recommendations
def show_recommendations():
    selected_user_id = user_listbox.get(tk.ACTIVE)
    if selected_user_id in user_item_matrix.index:
        recommendations = get_item_based_recommendations(selected_user_id, user_item_matrix, item_similarity_df, n=10)
        for item in tree.get_children():
            tree.delete(item)
        for idx, rec in enumerate(recommendations.index, 1):
            tree.insert("", "end", values=(idx, rec))
    else:
        messagebox.showerror("Error", f"User ID {selected_user_id} not found in the dataset")

# Nkriw l main window
root = tk.Tk()
root.title("Item-Based Recommendations")
root.geometry("600x500")
root.configure(bg='#E0F7FA')  # Light blue background color

# Napply styles
style = ttk.Style(root)
style.configure('TFrame', background='#E0F7FA')  # Light blue background for frames
style.configure('TLabel', font=('Helvetica', 12), foreground='#333', background='#E0F7FA')
style.configure('TButton', font=('Helvetica', 12, 'bold'), background='#0288D1', foreground='black')
style.map('TButton', background=[('active', '#0277BD')])
style.configure('Treeview.Heading', font=('Helvetica', 12, 'bold'), background='#0288D1', foreground='white')
style.configure('Treeview', font=('Helvetica', 12), rowheight=25, background='#E1F5FE', fieldbackground='#E1F5FE', foreground='#333')
style.map('Treeview', background=[('selected', '#B3E5FC')], foreground=[('selected', 'black')])

# Layout
main_frame = ttk.Frame(root, padding="10", style="TFrame")
main_frame.pack(fill=tk.BOTH, expand=True)

# User selection frame
user_frame = ttk.Frame(main_frame, style="TFrame")
user_frame.pack(pady=10)

ttk.Label(user_frame, text="Select User ID:", style="TLabel").pack()

user_listbox = tk.Listbox(user_frame, height=15, font=('Helvetica', 12), bg='#B3E5FC', fg='#333')
user_listbox.pack(pady=5)

for user_id in user_item_matrix.index[:50]:
    user_listbox.insert(tk.END, user_id)

# Button frame
button_frame = ttk.Frame(main_frame, style="TFrame")
button_frame.pack(pady=10)

recommend_button = ttk.Button(button_frame, text="Get Recommendations", command=show_recommendations, style="TButton")
recommend_button.pack()

# Recommendations frame
recommend_frame = ttk.Frame(main_frame, style="TFrame")
recommend_frame.pack(pady=10, fill=tk.BOTH, expand=True)

tree = ttk.Treeview(recommend_frame, columns=('Rank', 'Product ID'), show='headings', height=10, style="Treeview")
tree.heading('Rank', text='Rank')
tree.heading('Product ID', text='Product ID')
tree.column('Rank', width=100, anchor='center')
tree.column('Product ID', anchor='center')
tree.pack(fill=tk.BOTH, expand=True)

# Nbedo main event loop
root.mainloop()
