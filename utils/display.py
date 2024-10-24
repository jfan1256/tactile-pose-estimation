import time

# Print Header
def print_header(title, char='*', total_length=120):
    title = f'  {title}  '
    title_length = len(title)
    asterisks_length = (total_length - title_length) // 2
    header = f'{char}' * asterisks_length + title + f'{char}' * asterisks_length
    if len(header) < total_length:
        header += f'{char}'
    print(header + '\n')
    time.sleep(1)