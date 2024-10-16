
import streamlit as st
from pages import *

def main():

    st.write("# TxGraffiti")

    st.markdown(
        """
    *TxGraffiti* is an artificial intelligence designed to generate potential research conjectures
    in graph theory. Developed using Python and inspired by Fajtlowicz's GRAFFITI and DeLaVina's
    Graffiti.pc, TxGraffiti leverages machine learning and fine-tuned heuristics to identify
    significant relationships within mathematical data. By systematically calculating numerical
    properties and testing for potential inequalities, TxGraffiti highlights the most promising
    conjectures for further exploration.

    ## Terms of Usage
    I do not ask to be on any papers that result from using TxGraffiti. Still, I would appreciate a
    mention in the acknowledgments and a citation given to the TxGraffiti website. I am always happy
    to hear about the results generated by using TxGraffiti or to work collaboratively with you, so
    please feel free to contact me at rrd6@rice.edu if you wish to discuss anything further. Also,
    in the future, I will be adding a feature to the website where you can submit your
    counter-examples directly to the database of graphs from the website, and also submit to the website
    proven conjectures of the program.

    ### Contact
    - TxGraffiti Creator: Randy R. Davila, PhD

    - Email: rrd6@rice.edu and randyrdavila@gmail.com

        """
    )



if __name__ == "__main__":
    main()
