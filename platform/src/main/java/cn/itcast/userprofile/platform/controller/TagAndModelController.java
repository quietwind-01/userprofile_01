package cn.itcast.userprofile.platform.controller;


//import cn.itcast.up.common.HdfsTools;

import cn.itcast.up.common.HDFSUtils;
import cn.itcast.userprofile.platform.bean.Codes;
import cn.itcast.userprofile.platform.bean.HttpResult;
import cn.itcast.userprofile.platform.bean.dto.ModelDto;
import cn.itcast.userprofile.platform.bean.dto.TagDto;
import cn.itcast.userprofile.platform.bean.dto.TagModelDto;
import cn.itcast.userprofile.platform.service.TagService;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.UUID;

@RestController
public class TagAndModelController {
    @Autowired
    private TagService tagService;

    //实现1/2/3级标签添加
    /*
    PUT http://localhost:8081/tags/relation
    Content-Type: application/json

    [
      {
        "name": "Hello",
        "level": "1"
      },
      {
        "name": "Hello",
        "level": "2"
      },
      {
        "name": "Hello",
        "level": "3"
      }
    ]
     */
    @PutMapping("tags/relation")
    public void addTag(@RequestBody List<TagDto> tags) {
        tagService.saveTags(tags);
    }


    //实现1/2/3级标签显示
    //Get http://localhost:8081/tags?pid=-1
    @GetMapping("tags")
    public HttpResult<List<TagDto>> findByPidOrLevel(
            @RequestParam(required = false) Long pid,
            @RequestParam(required = false) Integer level) {

        List<TagDto> list = null;
        //如果传过来的是pid,那么直接用pid查询
        if(pid != null){
            list = tagService.findByPid(pid);
        }
        //如果传过来的是level,那么直接用level查询
        if(level != null){
            list = tagService.findByLevel(level);
        }

        return new HttpResult<List<TagDto>>(Codes.SUCCESS,"查询成功",list);
    }


    //4级标签添加
    @PutMapping("tags/model")
    public HttpResult putTagAndModel(@RequestBody TagModelDto tagModelDto){
        tagService.addTagModel(tagModelDto.getTag(),tagModelDto.getModel());
        return new HttpResult(Codes.SUCCESS,"成功",null);
    }

    //文件上传
    /**
     * 接收浏览器端上传的jar包,并将jar包上传到HDFS中,最后将存储路径返回给浏览器,因为后续浏览器端保存的时候需要将路径传递给
     * putTagAndModel方法将路径存储到model表的path字段中
     * @param file 是SpringMVC提供的用来接收文件的对象
     * @return
     */
    @PostMapping("tags/upload")
    public HttpResult<String> postTagFile(@RequestParam("file") MultipartFile file){
        //1.接收浏览器上传的文件并指定存储的路径
        String basePath = "hdfs://bd001:8020/temp/jars/";
        String fileName = UUID.randomUUID().toString() + ".jar"; //随机生成一个文件名称
        String path = basePath + fileName; //hdfs://bd001:8020/temp/jars/xxxxxx.jar

        //2.将文件上传到HDFS
        try {
            //注意:如果直接将file传递到hdfs,那么传上去的文件大小为0KB
            //所以可以这样:先将file保存到SpringBoot服务器中(Tomcat),名称任意,如temp.jar
            InputStream inputStream = file.getInputStream();
            IOUtils.copy(inputStream, new FileOutputStream(new File("temp.jar")));
            //再将服务器上的temp.jar上传到HDFS的指定的path
            HDFSUtils.getInstance().copyFromFile("temp.jar",path);
            System.out.println("==>>>>>>jar包已上传到hdfs的指定的path:\n"+path);
            return new HttpResult<>(Codes.SUCCESS,"文件上传成功",path);
        } catch (IOException e) {
            e.printStackTrace();
            return new HttpResult<>(Codes.ERROR,"文件上传失败","");
        }
    }


    //web平台整合Oozie
    @PostMapping("tags/{id}/model")
    public HttpResult changeModelState(@PathVariable Long id, @RequestBody ModelDto modelDto){
        //根据model的id修改model的状态
        tagService.updateModelState(id,modelDto.getState());
        return new HttpResult(Codes.SUCCESS,"启动成功",null);
    }


    /**
     * 4级标签查询
     *
     * @param pid
     * @return
     */
    @GetMapping("tags/model")
    public HttpResult getModel(Long pid) {
        List<TagModelDto> dto = tagService.findModelByPid(pid);
        return new HttpResult(Codes.SUCCESS, "查询成功", dto);
    }

    /**
     * 5级标签新增
     *
     * @param tagDto
     * @return
     */
    @PutMapping("tags/data")
    public HttpResult putData(@RequestBody TagDto tagDto) {
        tagService.addDataTag(tagDto);
        return new HttpResult(Codes.SUCCESS, "添加成功", null);
    }

}
