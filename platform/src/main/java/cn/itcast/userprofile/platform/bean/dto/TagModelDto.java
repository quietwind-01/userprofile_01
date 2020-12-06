package cn.itcast.userprofile.platform.bean.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class TagModelDto {
    private TagDto tag;
    private ModelDto model;
}