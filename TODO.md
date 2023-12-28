- Write down learnings from first training (see generation.ipynb, write down GPU used, hours taken etc., mention approach to prompting i.e. GPT-4 and sampling choices)

- Delete redundant clusters
    Before deleting cluster:
        - Download any local, uncommited files e.g. notebooks, models (?), text files

- Continue transcribing videos from Jamie Carragher - find out where to put lanugage='en' 

- Fix process_transcripts.py
    - nunique > 3 - can we aggregate?
    - seems like matching not working if 'No 1' etc. in title
- Check first line. If only a part sentence, drop podcast

- Data prep stuff below:


- Currently model is reliant on world knowledge of the guest to produce good responses. To add more context we could:
    - Add wiki page of guest at start or as system prompt (see https://huggingface.co/datasets/olm/wikipedia)
    - Generate wiki-style docs from what the guest said in podcast using GPT-4 (otherwise contents of wiki may not include everything about the guest)




- Mistral instruction-tuned
    - Using chat format Steven would be the "assistant" but does this lead to worse performance because typically the "user" is the one who asks questions. 
    - Yet we can't make Steven the "user" as we want the guest responses to vary depending on the person, and hence the guest needs to be the user
    - Compare against fine-tuning base model from scratch where model will learn new assistant/user interaction i.e. assistant asks the questions

- Create a Steven Bartlett LLM i.e. as if you were speaking to him (use Open AI speech-to-text) (?)