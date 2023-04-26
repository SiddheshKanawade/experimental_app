import streamlit as st
    # generate the dataset
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
    #%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from IPython.display import display
import time 
import runpy
import pickle
import os
from sklearn.pipeline import make_pipeline


st.set_page_config(
        page_title="Multipage App",
        page_icon="👋",
    )



st.title("Logistic Regression")
st.subheader("Forward and Backward Feature Selection")
st.header("Machine Learning")
st.markdown("---  ")
    #st.text("hello world")

    #st.markdown("**hello** World ")



bar = st.progress(0)



X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                            n_redundant=2, n_repeated=0, n_classes=2,
                            class_sep=2.0, random_state=42)

    # create a sequential forward feature selector
#sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=10)

    # create a sequential backward feature selector
#sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=10)

    # initialize empty lists to store the number of selected features and corresponding scores
n_features_fwd = []
scores_fwd = []
n_features_bwd = []
scores_bwd = []

st.write('Starting a long computation...')
    # loop through a range of iterations to select a variable number of features

lr = LogisticRegression()
lr.fit(X, y)



# create a sequential forward feature selector
# sfs = SequentialFeatureSelector(lr, direction='forward', n_features_to_select='auto')
# #sfs.fit(X, y)

# # create a sequential backward feature selector
# sbs = SequentialFeatureSelector(lr, direction='backward', n_features_to_select='auto')
# sbs.fit(X, y)

st.balloons()
    #st.write('...and now we\'re done!')
st.success('Done!')

    # define a function to plot the scores
@st.cache_data
def plot_scores(n):
        fig=plt.figure()
        plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
        plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
        plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Sequential Feature Selection')
        plt.legend()
        #plt.show()
        st.pyplot(fig)

    




if __name__ == '__main__':
        #run_app()   

    # create a slider for the number of iterations
        iterations_slider = st.slider(label="Iterations", min_value=1, max_value=len(n_features_fwd), value=len(n_features_fwd))

    # use the interact function to link the slider to the plot
        interact(plot_scores, n=iterations_slider)
        for i in range(1, iterations_slider):

            # create a new instance of the logistic regression model
            sfs.n_features_to_select = i
            sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=i) 
            # fit the sequential forward feature selector
            sfs.fit(X, y)
            # get the selected features and their corresponding scores
            selected_features_fwd = sfs.transform(X)
            lr.fit(selected_features_fwd, y)
            score_fwd = accuracy_score(y, lr.predict(selected_features_fwd))
            # append the number of selected features and the corresponding score to the lists
            n_features_fwd.append(i)
            scores_fwd.append(score_fwd)
            st.write(score_fwd)



            #time.sleep(0.05)


            # create a new instance of the logistic regression model

            sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=i)
            # fit the sequential backward feature selector
            sbs.n_features_to_select = i

            # get the selected features and their corresponding scores
            selected_features_bwd = sbs.transform(X)
            sbs.fit(X, y)
            lr.fit(selected_features_bwd, y)
            score_bwd = accuracy_score(y, lr.predict(selected_features_bwd))
            # append the number of selected features and the corresponding score to the lists
            n_features_bwd.append(i)
            scores_bwd.append(score_bwd)
            st.write(score_bwd)
            
            


            bar.progress(i+20)

