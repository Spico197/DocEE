from rex.utils.io import load_jsonlines

if __name__ == "__main__":
    data = load_jsonlines("Data/DuEEData/duee_fin_train.json")
    for d in data:
        if len(d["event_list"]) > 1:
            triggers = set(record["trigger"] for record in d["event_list"])
            if len(triggers) < 2:
                print(d)
                break

    {
        "id": "e55ffdd8af2e77df58b3020c8ee5502a",
        "title": "苹果已收购企业设备管理公司Fleetsmith",
        "text": "原标题：苹果已收购企业设备管理公司Fleetsmith    来源：威锋网\n苹果已经收购了企业设备管理公司 Fleetsmith，该公司于周三宣布了这一消息。\n成立于 2014 年的 Fleetsmith 在博客文章中宣布他们现在已成为苹果的一员，并称他们“很高兴加入苹果”。该公司为 IT 部门提供企业解决方案，以管理 Mac，iPad 和 iPhone。\n该公司写道：“我们的共同价值观是在不牺牲隐私和安全的前提下将客户放在我们所做一切工作的中心，这意味着我们可以真正实现我们的使命，将 Fleetsmith 提供给世界各地各种规模的企业和机构。”\n在收购的同一周，苹果在 WWDC 2020 举行了“管理苹果设备的新功能”开发人员会议。在会议期间，苹果宣布了 Mac Pro 的新管理功能，Mac Supervision 的更改以及 macOS Big Sur 中的托管软件更新以及其它功能。\n通过 Fleetsmith 的收购，苹果可能希望进一步为企业和教育客户增强其第一方设备管理选项。到目前为止，苹果公司主要依靠第三方解决方案为其客户提供 MDM 平台。\nFleetsmith 是苹果在 2020 年的又一笔收购，之前的收购包括天气应用 Dark Sky，实时 VR 活动公司 NextVR 和 AI 初创公司 Voysis。",
        "event_list": [
            {
                "trigger": "收购",
                "event_type": "企业收购",
                "arguments": [
                    {"role": "收购方", "argument": "苹果"},
                    {"role": "被收购方", "argument": "Fleetsmith"},
                    {"role": "披露时间", "argument": "周三"},
                ],
            },
            {
                "trigger": "收购",
                "event_type": "企业收购",
                "arguments": [
                    {"role": "收购方", "argument": "苹果"},
                    {"role": "被收购方", "argument": "Dark Sky"},
                    {"role": "被收购方", "argument": "NextVR"},
                    {"role": "被收购方", "argument": "Voysis"},
                    {"role": "收购完成时间", "argument": "2020 年"},
                ],
            },
        ],
    }
