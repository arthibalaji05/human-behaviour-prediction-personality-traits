"""
Project: Predicting Human Behaviour with Human Traits

Description:
Machine learning project analysing personality traits using
PCA dimensionality reduction and K-Means clustering.

Techniques:
- Data preprocessing
- Exploratory data analysis
- PCA
- K-Means clustering
- Tkinter GUI

Author:
Arthi Balaji
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import messagebox
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick


data = pd.read_csv('data-final.csv', delimiter='\t')  
data.dropna(inplace=True)
continents = pd.read_csv('continents2.csv')

data1 = data.copy()
data1['alpha-2'] = data1['country']
continents = data1.merge(continents, on=["alpha-2"], how='left')

from mpl_toolkits.mplot3d import Axes3D

def perform_clustering():
    try:
        # Drop rows with NaN in region/continent-related columns
        continents_cleaned = continents.dropna(subset=['region'])

        # Group dataset by regions/continents
        grouped = continents_cleaned.groupby('region')

        all_data = []
        all_regions = []

        for region, group in grouped:
            # Extract personality-related columns
            question_columns = [col for col in group.columns if col.startswith(('EXT', 'AGR', 'CSN', 'OPN', 'EST'))]

            # Select data and handle missing values
            data_selected = group[question_columns].dropna()

            if data_selected.empty:
                print(f"No valid data for region: {region}")
                continue

            all_data.append(data_selected)
            all_regions.extend([region] * len(data_selected))

        # Combine all regions' data
        combined_data = pd.concat(all_data)

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)

        # PCA for 2D visualization
        pca = PCA(n_components=2)  # Keep only 2 components for visualization
        pca_data = pca.fit_transform(scaled_data)

        # K-Means clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=0)
        labels = kmeans.fit_predict(pca_data)

        # Create DataFrame for visualization
        df_pca = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
        df_pca['Clusters'] = labels
        df_pca['Region'] = all_regions  # Assign corresponding region

        # Plot 2D PCA visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='tab10', alpha=0.8, edgecolor="black")
        plt.title('Personality Clusters (PCA 2D Projection) - 5 Clusters', fontsize=16)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()

    except Exception as e:
        print(f"Error during clustering: {str(e)}")



def plot_personality_traits(traits, title):
    plt.figure(figsize=[15, 15])
    for n, f in enumerate(traits, start=1):
        plt.subplot(5, 2, n)
        sns.countplot(x=f, edgecolor="black", alpha=0.7, data=data)
        plt.title(f"{title}: {f}")
    plt.tight_layout()
    plt.show()


print("Merged Continents DataFrame:")
print(continents.head())  
print("Unique regions in merged data:", continents['region'].unique())

def plot_continental_personality_traits(trait_list, title, trait_labels=None):
    df = continents.groupby(["region"])[trait_list].count()  
    if trait_labels:
        df.columns = trait_labels  # Apply trait labels if provided
    else:
        df.columns = trait_list  # Keep original column names for other graphs

    df = df.loc[['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'], :]
    df = df.T  

    df["World"] = df.sum(axis=1)  
    df = df.sort_values(by="World", ascending=True)
    df["Min"] = df.min(axis=1)
    df["Max"] = df.max(axis=1)

    # Setting up figure and axes
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 1)
    gs.update(wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])

    # Change background color
    background_color = "#fbfbfb"
    fig.patch.set_facecolor(background_color)  
    ax0.set_facecolor(background_color)  

    y_dummy = np.arange(1, len(df.index) + 1)

    ax0.hlines(y=y_dummy, xmin=df["Min"], xmax=df["Max"], color='grey', alpha=0.4, zorder=3)
    ax0.scatter(df['World'], y_dummy, color='red', label='World')
    ax0.scatter(df['Africa'], y_dummy, color='green', label='Africa')
    ax0.scatter(df['Americas'], y_dummy, color='blue', label='Americas')
    ax0.scatter(df['Asia'], y_dummy, color='orange', label='Asia')
    ax0.scatter(df['Europe'], y_dummy, color='skyblue', label='Europe')
    ax0.scatter(df['Oceania'], y_dummy, color='magenta', label='Oceania')

    y_label = list(df.index)
    y_label.insert(0, "")  
    ax0.yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax0.set_yticklabels(y_label)
    ax0.set_xticklabels([])
    ax0.tick_params(bottom=False)

    # Title placement
    ax0.text(-100, len(y_dummy) + 1, title, fontsize=16, fontweight='bold', fontfamily='serif')

    ax0.legend(loc='lower center', ncol=7, bbox_to_anchor=(0.53, -0.1))

    # Hide borders
    for s in ["top", "right", "left", "bottom"]:
        ax0.spines[s].set_visible(False)

    plt.show()



def show_extroversion():
    plot_personality_traits(['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'], "Extroversion Q&As")

def show_neuroticism():
    plot_personality_traits(['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'], "Neuroticism Q&As")

def show_agreeableness():
    plot_personality_traits(['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'], "Agreeableness Q&As")

def show_conscientiousness():
    plot_personality_traits(['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'], "Conscientiousness Q&As")

def show_openness():
    plot_personality_traits(['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'], "Openness Q&As")

def show_continental_extroversion():
    plot_continental_personality_traits(
        ['EXT1','EXT2','EXT3','EXT4','EXT5','EXT6','EXT7','EXT8','EXT9','EXT10'],
        'Visualization of Extraversion across Continents',
        ['I am the life of the party', 'I don’t talk a lot', 'I feel comfortable around people',
         'I keep in the background', 'I start conversations', 'I have little to say',
         'I talk to a lot of different people at parties', 'I don’t like to draw attention to myself',
         'I don’t mind being the center of attention', 'I am quiet around strangers']
    )

def show_continental_neuroticism():
    plot_continental_personality_traits(
        ['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'],
        'Visualization of Neuroticism across Continents',
        ['I get stressed out easily', 'I am relaxed most of the time', 'I worry about things',
         'I seldom feel blue', 'I am easily disturbed', 'I get upset easily',
         'I change my mood a lot', 'I have frequent mood swings', 'I get irritated easily',
         'I often feel blue']
    )


def show_continental_agreeableness():
    plot_continental_personality_traits(
        ['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'],
        'Visualization of Agreeableness across Continents',
        ['I feel little concern for others', 'I am interested in people', 'I insult people',
         'I sympathize with others’ feelings', 'I am not interested in other people’s problems',
         'I have a soft heart', 'I am not really interested in others', 'I take time out for others',
         'I feel others’ emotions', 'I make people feel at ease']
    )

def show_continental_conscientiousness():
    plot_continental_personality_traits(
        ['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'],
        'Visualization of Conscientiousness across Continents',
        ['I am always prepared', 'I leave my belongings around',
         'I pay attention to details', 'I make a mess of things',
         'I get chores done right away', 'I often forget to put things back in their proper place',
         'I like order', 'I shirk my duties',
         'I follow a schedule', 'I am exacting in my work']
    )


def show_continental_openness():
    plot_continental_personality_traits(
        ['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'],
        'Visualization of Openness to Experience across Continents',
        ['I have a rich vocabulary', 'I have difficulty understanding abstract ideas',
         'I have a vivid imagination', 'I am not interested in abstract ideas',
         'I have excellent ideas', 'I do not have a good imagination',
         'I am quick to understand things', 'I use difficult words',
         'I spend time reflecting on things', 'I am full of ideas']
    )


def predict_personality():
    questions = [
        'I am the life of the party (Yes/No)',
        'I get stressed out easily (Yes/No)',
        'I feel little concern for others (Yes/No)',
        'I am always prepared (Yes/No)',
        'I have a rich vocabulary (Yes/No)'
    ]
   
    # Initialize response storage
    responses = []

    def get_response():
        response = response_var.get()
        if response not in ['Yes', 'No']:
            messagebox.showerror("Invalid Input", "Please enter 'Yes' or 'No'")
            return

        responses.append(1 if response == 'Yes' else 0)
        if len(responses) < len(questions):
            question_label.config(text=questions[len(responses)])
            response_var.set("")
        else:
            calculate_and_display_results()
            response_window.destroy()

    def calculate_and_display_results():
        traits = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        counts = np.array(responses)  
        percentages = (counts / len(questions)) * 100


        plt.figure(figsize=(8, 8))
        plt.pie(percentages, labels=traits, autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Personality Traits Percentage')
        plt.axis('equal')  
        plt.show()

   
        final_predicted_trait = traits[np.argmax(percentages)]
        final_label.config(text=f"Final Predicted Trait: {final_predicted_trait}")

   
    response_window = Toplevel(root)
    response_window.title("Predict Personality")

    response_var = StringVar()
    question_label = Label(response_window, text=questions[0], font=("Arial", 14))
    question_label.pack(pady=20)

    entry = Entry(response_window, textvariable=response_var, font=("Arial", 14))
    entry.pack(pady=20)
    entry.bind("<Return>", lambda event: get_response())

    next_button = Button(response_window, text="Next", command=get_response)
    next_button.pack(pady=10)

    final_label = Label(response_window, text="", font=("Arial", 14))
    final_label.pack(pady=20)

def create_gui():
    global root
    root = Tk()
    root.title("Personality Traits Analysis")

    frame = Frame(root)
    frame.pack(pady=20)

    buttons = [
        ("Extroversion", show_extroversion),
        ("Neuroticism", show_neuroticism),
        ("Agreeableness", show_agreeableness),
        ("Conscientiousness", show_conscientiousness),
        ("Openness", show_openness),
        ("Continental Extraversion", show_continental_extroversion),
        ("Continental Neuroticism", show_continental_neuroticism),
        ("Continental Agreeableness", show_continental_agreeableness),
        ("Continental Conscientiousness", show_continental_conscientiousness),
        ("Continental Openness", show_continental_openness),
        ("Predict Personality", predict_personality),
        ("Perform Clustering", perform_clustering)
    ]

    for (text, command) in buttons:
        Button(frame, text=text, command=command).pack(pady=5)

    root.mainloop()

create_gui()
