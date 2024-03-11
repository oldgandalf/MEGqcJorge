# import inquirer
# questions = [
#   inquirer.Checkbox('interests',
#                     message="What are you interested in?",
#                     choices=['Computers', 'Books', 'Science', 'Nature', 'Fantasy', 'History'],
#                     ),
# ]
# answers = inquirer.prompt(questions)

# print(answers["interests"])


# from prompt_toolkit.shortcuts import checkboxlist_dialog
# from prompt_toolkit.styles import Style

# results = checkboxlist_dialog(
#     title="CheckboxList dialog",
#     text="What would you like in your breakfast ?",
#     values=[
#         ("eggs", "Eggs"),
#         ("bacon", "Bacon"),
#         ("croissants", "20 Croissants"),
#         ("daily", "The breakfast of the day")
#     ],
#     style=Style.from_dict({
#         'dialog': 'bg:#cdbbb3',
#         'button': 'bg:#bf99a4',
#         'checkbox': '#e8612c',
#         'dialog.body': 'bg:#a9cfd0',
#         'dialog shadow': 'bg:#c98982',
#         'frame.label': '#fcaca3',
#         'dialog.body label': '#fd8bb6',
#     })
# ).run()

# print(results)


import questionary

# Define the options
options = ['Option 1', 'Option 2', 'Option 3']

# Ask the user to check some options
checked_options = questionary.checkbox("Select options:", choices=options).ask()

print('You checked:', checked_options)