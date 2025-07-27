import os
import json
import re

def parse_fund(text):
    # Basic fund-level extraction using regex
    fund = {}
    fund["fund_name"] = re.search(r"#\s*(.*)", text).group(1) if re.search(r"#\s*(.*)", text) else "Unknown"
    fund["location"] = re.search(r"Location\s*\n(.*)", text).group(1).strip() if re.search(r"Location\s*\n(.*)", text) else None
    fund["website"] = re.search(r"Fund Website\]\((.*?)\)", text).group(1) if re.search(r"Fund Website\]\((.*?)\)", text) else None
    fund["contact_method"] = re.search(r"Best way to get in touch\s*\n(.*)", text)
    fund["contact_method"] = fund["contact_method"].group(1).strip() if fund["contact_method"] else "Unknown"
    fund["deck_submission_link"] = re.search(r"Submit a Deck\]\((.*?)\)", text).group(1) if re.search(r"Submit a Deck\]\((.*?)\)", text) else None
    fund["check_size_ranges"] = re.findall(r"\$\d.*", text)
    fund["investment_rounds"] = re.findall(r"(Pre-Seed|Seed|Series A|Series B\+)", text)
    fund["lead_rounds"] = re.findall(r"Rounds they lead\s*\n([^\n]+)", text)
    fund["sectors"] = re.findall(r"Sectors they invest in\s*\n(.*?)\n", text)
    fund["geographies"] = re.findall(r"Geographies they invest in\s*\n(.*?)\n", text)
    fund["unicorn_investments"] = re.findall(r"\[([^\]]+)\]\(", text)  # Grabs names inside [links]
    return fund

def parse_individuals(text):
    # Extract VC partner block details
    individuals = []
    blocks = text.split("View Profile")
    for block in blocks:
        name_match = re.search(r"(\w+[\s\w\.]+)\n(Managing Director|General Partner|Co-Founding Partner|Founding Partner|Partner|Principal)", block)
        if not name_match:
            continue
        person = {
            "name": name_match.group(1).strip(),
            "role": name_match.group(2).strip(),
            "affiliation": re.search(r"#\s*(.*)", text).group(1) if re.search(r"#\s*(.*)", text) else "Unknown",
            "location": re.search(r"Location\s*\n(.*)", text).group(1).strip() if re.search(r"Location\s*\n(.*)", text) else None,
            "contact_links": {
                "twitter": re.search(r"\(https:\/\/twitter\.com\/[^\)]+\)", block),
                "linkedin": re.search(r"\(https:\/\/www\.linkedin\.com\/[^\)]+\)", block),
                "profile": re.search(r"\(https:\/\/www\.vcsheet\.com\/who\/[^\)]+\)", block)
            },
            "dm_open": True if "DMs open" in block else False
        }
        # Clean URLs
        for key in person["contact_links"]:
            match = person["contact_links"][key]
            person["contact_links"][key] = match.group(0)[1:-1] if match else None
        individuals.append(person)
    return individuals

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    fund_json = parse_fund(content)
    vc_jsons = parse_individuals(content)
    return fund_json, vc_jsons

def main(folder_path):
    all_funds = []
    all_vcs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            fund, vcs = process_file(os.path.join(folder_path, file))
            all_funds.append(fund)
            all_vcs.extend(vcs)
    
    with open("vc_funds.json", "w", encoding="utf-8") as f_fund:
        json.dump(all_funds, f_fund, indent=2)

    with open("vc_individuals.json", "w", encoding="utf-8") as f_vc:
        json.dump(all_vcs, f_vc, indent=2)

    print("✅ Parsed and saved vc_funds.json + vc_individuals.json")

if __name__ == "__main__":
    main(r"D:\Desktop\New folder (4)\funds")
