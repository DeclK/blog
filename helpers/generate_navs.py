""" Automatically generate some navigations for mkdocs:
1. links of each Category
2. check `blogs` key in mkdocs.yml
"""
from pathlib import Path
import re
from mkdocs.utils import yaml_load

this_dir = Path(__file__).parent
root_dir = this_dir.parent
doc_dir = root_dir / "docs"
yml_file = root_dir / "mkdocs.yml"

exclude_dir = set(["assets", "css", "js"])

fixed_front_matter = """---
icon: material/dots-grid
comments: false
nostatistics: true
---"""
def print_red(str):
    print("\033[1;31;40m" + str + "\033[0m")

def print_green(str):
    print("\033[1;32;40m" + str + "\033[0m")

def get_blog_dirs_and_index_md():
    blog_dirs = []
    index_md = []
    for path in doc_dir.iterdir():
        if path.is_dir():
            if path.name in exclude_dir:
                continue
            blog_dirs.append(path)
            index_md.append(path / "index.md")
    return blog_dirs, index_md


def get_front_matter_text(text: str) -> str | None:
    m = re.match(r'(?s)^---\n(.*?)\n---', text)
    if m:
        return re.sub(r'(?s)^---\n.*?\n---', fixed_front_matter, text)
    else:
        return fixed_front_matter + "\n" + text


def sort_and_pop_index_md(md_list: list):
    md_list.sort(key=lambda x: x.name)
    ret_list = []
    for md in md_list:
        if md.name == "index.md": continue
        ret_list.append(md)
    return ret_list


def format_front_matter():
    """ Make sure front matter of index.md is fixed. """
    _, index_md = get_blog_dirs_and_index_md()
    for md in index_md:
        text = md.read_text(encoding="utf-8")
        # write format front matter into md
        with open(md, "w", encoding="utf-8") as f:
            f.write(get_front_matter_text(text))


def check_blogs_dirs_to_mkdocs_yml():
    # load yaml
    with open(yml_file, encoding="utf-8") as f:
        mkdocs_yaml = yaml_load(f)
    # get blog dirs
    plugin_list = mkdocs_yaml["plugins"]
    for plugin_dict in plugin_list:
        if "blogging" in plugin_dict:
            blog_dir = plugin_dict["blogging"]["dirs"]
            expect_blog_path, _ = get_blog_dirs_and_index_md()
            expect_blog_name = [p.name for p in expect_blog_path]

            if set(blog_dir) != set(expect_blog_name):
                print_red("mkdocs.yml blog_dirs is not equal to docs dirs!!")
                print("mkdocs.yml:", blog_dir)
                print("expected:", expect_blog_name)
            else:
                print_green("mkdocs.yml blog_dirs is expected")


def generate_links_for_category():
    _, index_mds = get_blog_dirs_and_index_md()
    for idx_md in index_mds:
        idx_md_text = idx_md.read_text(encoding="utf-8")
        idx_dir = idx_md.parent
        md_files = list(idx_dir.rglob("*.md"))
        md_files = sort_and_pop_index_md(md_files)
        nav_lines = ["    - [{}]({})".format(p.stem, p.relative_to(idx_dir).as_posix())
                     for p in md_files]
        pattern = re.compile(r'!!! abstract "Table of Contents".*', re.DOTALL)

        postfix_line = "!!! abstract \"Table of Contents\""

        if not pattern.search(idx_md_text):
            # force to add the table of content
            print_red("[WARNING] No table of content found in index.md.")
            idx_md_text = idx_md_text.rstrip()
            idx_md_text += "\n" * 2 + postfix_line + "\n"

        new_content = postfix_line + "\n" + "\n".join(nav_lines)
        new_text = pattern.sub(new_content, idx_md_text)
        # print(new_text)
        with open(idx_md, "w", encoding="utf-8") as f:
            f.write(new_text)

            
if __name__ == "__main__":
    format_front_matter()
    check_blogs_dirs_to_mkdocs_yml()
    generate_links_for_category()