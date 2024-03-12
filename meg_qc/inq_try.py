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

# Define the categories and subcategories
categories = {
    'sub': {'029', '026', '025', '021', '016', '028', '022', '030', '018', '020', '009', '015', '013', '027', '014', '017', '012', '019', '023', '024', '031'},
    'ses': {'1'},
    'task': {'induction', 'deduction'},
    'run': {1},
    'desc': {'REPORT', 'ECGs', 'EOGs', 'Muscle', 'epochs', 'Sensors', 'PSDs', 'STDs', 'SimpleMetrics'}
}

# Create a list of subcategories with category titles
subcategories = []
for category, items in categories.items():
    subcategories.append(questionary.Separator(f'== {category} =='))
    for item in items:
        subcategories.append(str(item))

# Ask the user to select a subcategory
subcategory = questionary.select("Select a subcategory:", choices=subcategories).ask()

print('You selected:', subcategory)