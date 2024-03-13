# import questionary

# # Define the categories and subcategories
# categories = {
#     'sub': {'029', '026', '025', '021', '016', '028', '022', '030', '018', '020', '009', '015', '013', '027', '014', '017', '012', '019', '023', '024', '031'},
#     'ses': {'1'},
#     'task': {'induction', 'deduction'},
#     'run': {1},
#     'desc': {'REPORT', 'ECGs', 'EOGs', 'Muscle', 'epochs', 'Sensors', 'PSDs', 'STDs', 'SimpleMetrics'}
# }

# # Create a list of subcategories with category titles
# subcategories = []
# for category, items in categories.items():
#     subcategories.append(questionary.Separator(f'== {category} =='))
#     for item in items:
#         subcategories.append(str(item))

# # Ask the user to select a subcategory
# subcategory = questionary.select("Select a subcategory:", choices=subcategories).ask()

# print('You selected:', subcategory)



from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style

# # Define the categories and subcategories
# categories = {
#     'METRIC': ['_ALL_metrics_', 'ECGs', 'EOGs', 'Muscle', 'PSDs', 'STDs'],
#     'SUBJECT': ['_ALL_subjects_', '029', '026', '025', '021', '016', '028', '022', '030', '018', '020', '009', '015', '013', '027', '014', '017', '012', '019', '023', '024', '031'],
#     'SESSION': ['_ALL_sessions_', '1'],
#     'TASK': ['_ALL_tasks_', 'induction', 'deduction'],
#     'RUN': ['_ALL_runs_', 1]
# }

# # Create a list of values with category titles
# values = []
# for category, items in categories.items():
#     values.append((f'== {category} ==', f'== {category} =='))
#     for item in items:
#         values.append((str(item), str(item)))

# results = checkboxlist_dialog(
#     title="CheckboxList dialog",
#     text="Select subcategories:",
#     values=values,
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

# # Ignore the category titles
# selected_subcategories = [result for result in results if not result.startswith('== ')]

# print('You selected:', selected_subcategories)


def modify_categories(categories):

    old_new_categories = {'desc': 'METRIC', 'sub': 'SUBJECT', 'ses': 'SESSION', 'task': 'TASK', 'run': 'RUN'}

    categories_copy = categories.copy()
    for category, subcategories in categories_copy.items():
        # Convert the set of subcategories to a sorted list
        sorted_subcategories = sorted(subcategories, key=str)
        # If the category is in old_new_categories, replace it with the new category
        if category in old_new_categories:
            new_category = old_new_categories[category]
            categories[new_category] = categories.pop(category)
            # Replace the original set of subcategories with the modified list
            sorted_subcategories.insert(0, '_ALL_'+new_category+'S_')
            categories[new_category] = sorted_subcategories
            
    return categories

def selector(entities):

    # Define the categories and subcategories
    categories = modify_categories(entities)

    # Create a list of values with category titles
    values = []
    for category, items in categories.items():
        values.append((f'== {category} ==', f'== {category} =='))
        for item in items:
            values.append((str(item), str(item)))

    results = checkboxlist_dialog(
        title="CheckboxList dialog",
        text="Select subcategories:",
        values=values,
        style=Style.from_dict({
            'dialog': 'bg:#cdbbb3',
            'button': 'bg:#bf99a4',
            'checkbox': '#e8612c',
            'dialog.body': 'bg:#a9cfd0',
            'dialog shadow': 'bg:#c98982',
            'frame.label': '#fcaca3',
            'dialog.body label': '#fd8bb6',
        })
    ).run()

    # Ignore the category titles
    selected_subcategories = [result for result in results if not result.startswith('== ')]

    print('You selected:', selected_subcategories)

    return selected_subcategories

entities = {'sub': {'026', '031', '023', '021', '019', '020', '018', '013', '016', '030', '022', '027', '015', '025', '014', '029', '009', '017', '012', '028', '024'}, 'ses': {'1'}, 'task': {'induction', 'deduction'}, 'run': {1}, 'desc': {'PSDs', 'REPORT', 'SimpleMetrics', 'ECGs', 'STDs', 'epochs', 'Muscle', 'Sensors', 'EOGs'}}

selector(entities)