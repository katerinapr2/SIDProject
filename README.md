# SIDProject
A natural language processing model for finding sensitive information in Greek court decision documents. 

Try it out by installing it with

```julia
] add https://github.com/katerinapr2/SIDProject
```
and then issuing
```julia
using SIDProject
text = "Ο Γιώργος Σεφέρης ήταν Έλληνας διπλωμάτης και ποιητής και ο πρώτος Έλληνας που τιμήθηκε με Νόμπελ Λογοτεχνίας. Το πραγματικό του όνομα ήταν Γεώργιος Σεφεριάδης. Γεννήθηκε στα Βουρλά στις 29 Φεβρουαρίου του 1900 και ήταν ο πρωτότοκος γιος του Στέλιου και της Δέσπως (το γένος Γ. Τενεκίδη) Σεφεριάδη."
anonymize(text)
```
Then a txt file is created in which each sensitive information has been flagged. 

## Usage via Python 
```python
import julia
j = julia.Julia()
j.eval('using Pkg; Pkg.activate("SIDProject")')
SIDProject.anonymize(text)
```

[toolkit]: https://github.com/nlpaueb/gr-nlp-toolkit
[pycall]: https://github.com/JuliaPy/PyCall.jl
