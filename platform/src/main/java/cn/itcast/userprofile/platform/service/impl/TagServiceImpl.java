package cn.itcast.userprofile.platform.service.impl;

import cn.itcast.userprofile.platform.bean.dto.ModelDto;
import cn.itcast.userprofile.platform.bean.dto.TagDto;
import cn.itcast.userprofile.platform.bean.dto.TagModelDto;
import cn.itcast.userprofile.platform.bean.po.ModelPo;
import cn.itcast.userprofile.platform.bean.po.TagPo;
import cn.itcast.userprofile.platform.repo.ModelRepository;
import cn.itcast.userprofile.platform.repo.TagRepository;
import cn.itcast.userprofile.platform.service.Engine;
import cn.itcast.userprofile.platform.service.TagService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class TagServiceImpl implements TagService {
    @Autowired
    private TagRepository tagRepo;
    @Autowired
    private ModelRepository modelRepo;
    @Autowired
    private Engine engine;


    /**
     * 将前台传递过来的1/2/3级标签保存到数据库
     *
     * @param tags
     */
    @Override
    public void saveTags(List<TagDto> tags) {
        System.out.println(tags);
        //[TagDto(id=null, name=房地产, rule=null, level=1, pid=null),
        // TagDto(id=null, name=链家, rule=null, level=2, pid=null),
        // TagDto(id=null, name=人口属性, rule=null, level=3, pid=null)]
        //后续保存到数据库
        //注意:现在我手里的TagDto是传输对象,保存到数据库要的是TagPo持久化对象,所以要先转换
        //1.TagDto转换为TagPo
        TagPo tagPo1 = this.convert(tags.get(0));
        TagPo tagPo2 = this.convert(tags.get(1));
        TagPo tagPo3 = this.convert(tags.get(2));

        //定义一个变量存储id
        TagPo temp = null;

        //2.先保存1级标签
        TagPo tagResult = tagRepo.findByNameAndLevelAndPid(tagPo1.getName(), tagPo1.getLevel(), tagPo1.getPid());
        if (tagResult == null) {
            //没查到
            temp = tagRepo.save(tagPo1);//保存1级标签,保存完之后就有了id
        } else {
            //查到了
            temp = tagResult;
        }
        //走到这里temp就记录了1级标签的信息,包括id

        //3.将1级标签的id作为2级标签的pid,再保存2级标签
        TagPo tagResult2 = tagRepo.findByNameAndLevelAndPid(tagPo2.getName(), tagPo2.getLevel(), tagPo2.getPid());
        if (tagResult2 == null) {
            //将1级标签的id作为2级标签的pid,再保存2级标签
            tagPo2.setPid(temp.getId());
            temp = tagRepo.save(tagPo2);//把保存好的2级标签赋值给临时变量
        } else {
            temp = tagResult2;
        }
        //代码走到这里,temp中就记录了2级标签的信息,包括2级标签的id

        //4.将2级标签的id作为3级标签的pid,再保存3级标签
        TagPo tagResult3 = tagRepo.findByNameAndLevelAndPid(tagPo3.getName(), tagPo3.getLevel(), tagPo3.getPid());
        if (tagResult3 == null) {
            tagPo3.setPid(temp.getId());
            tagRepo.save(tagPo3);
        }

    }


    /**
     * 根据pid查询List<TagDto>
     *
     * @param pid
     * @return
     */
    @Override
    public List<TagDto> findByPid(Long pid) {
        List<TagPo> list = tagRepo.findByPid(pid);
        //注意:数据库查出来的是List<TagPo>,而要返回的是List<TagDto>,所以需要转换
        //方式1:使用java8之前的传统写法
        /*List<TagDto> newList = new ArrayList<>();
        for (TagPo tagPo : list) {
            TagDto tagDto = this.convert(tagPo);
            newList.add(tagDto);
        }*/

        //方式2:使用Java8的Lambda表达式和StreamAPI
        /*List<TagDto> newList = list.stream().map((po) -> {
            TagDto tagDto = this.convert(po);
            return tagDto;
        }).collect(Collectors.toList());*/

        //方式3:Java8简写
        List<TagDto> newList = list.stream()
                .map(this::convert)//行为参数化
                .collect(Collectors.toList());

        return newList;
    }

    @Override
    public List<TagDto> findByLevel(Integer level) {
        List<TagPo> list = tagRepo.findByLevel(level);
        List<TagDto> listDto = list.stream().map(this::convert).collect(Collectors.toList());
        return listDto;
    }


    @Override
    public void addTagModel(TagDto tagDto, ModelDto modelDto) {
        //保存tag
        //TagDto->TagPo
        TagPo tagPo = this.convert(tagDto);
        tagPo = tagRepo.save(tagPo);//把保存完的状态接收一下,因为后面要用的保存完的tag的id

        //保存model
        //ModelDto->ModelPo
        //注意:保存model的时候需要tag的id,所以上面保存完tag需要把保存完的状态接收一下
        ModelPo modelPo = this.convert(modelDto, tagPo.getId());
        modelRepo.save(modelPo);
    }

    @Override
    public List<TagModelDto> findModelByPid(Long pid) {
        List<TagPo> tagPos = tagRepo.findByPid(pid);
        return tagPos.stream().map((tagPo) -> {
            Long id = tagPo.getId();
            ModelPo modelPo = modelRepo.findByTagId(id);
            if (modelPo == null) {
                //找不到model,就只返回tag
                return new TagModelDto(convert(tagPo), null);
            }
            return new TagModelDto(convert(tagPo), convert(modelPo));
        }).collect(Collectors.toList());
    }

    @Override
    public void addDataTag(TagDto tagDto) {
        tagRepo.save(convert(tagDto));
    }


    //根据model的id修改model的状态
    //不仅仅是修改,还应该要真正的启动/停止任务
    @Override
    public void updateModelState(Long id, Integer state) {
        String jobId = null;
        ModelPo modelPo = modelRepo.findByTagId(id);
        if(state == 3){//启动
            //启动任务
            //调用engine类中的封装好的方法启动任务
            jobId = engine.startModel(this.convert(modelPo));
            modelPo.setName(jobId);
        }
        if(state == 4){//停止
            //停止任务
            //调用engine类中的封装好的方法停止任务
            engine.stopModel(this.convert(modelPo));
        }
        //更新状态
        modelPo.setState(state);
        modelRepo.save(modelPo);
    }


    private ModelDto convert(ModelPo modelPo) {
        ModelDto modelDto = new ModelDto();
        modelDto.setId(modelPo.getId());
        modelDto.setName(modelPo.getName());
        modelDto.setMainClass(modelPo.getMainClass());
        modelDto.setPath(modelPo.getPath());
        modelDto.setArgs(modelPo.getArgs());
        modelDto.setState(modelPo.getState());
        modelDto.setSchedule(modelDto.parseDate(modelPo.getSchedule()));
        return modelDto;
    }

    /**
     * modelDto转为modelPo
     *
     * @param modelDto
     * @param id
     * @return
     */
    private ModelPo convert(ModelDto modelDto, Long id) {
        ModelPo modelPo = new ModelPo();
        modelPo.setId(modelDto.getId());
        modelPo.setTagId(id);
        modelPo.setName(modelDto.getName());
        modelPo.setMainClass(modelDto.getMainClass());
        modelPo.setPath(modelDto.getPath());
        modelPo.setSchedule(modelDto.getSchedule().toPattern());
        modelPo.setCtime(new Date());
        modelPo.setUtime(new Date());
        modelPo.setState(modelDto.getState());
        modelPo.setArgs(modelDto.getArgs());
        return modelPo;
    }


    /**
     * po转换为dto对象
     *
     * @param tagPo
     * @return
     */
    private TagDto convert(TagPo tagPo) {
        TagDto tagDto = new TagDto();
        tagDto.setId(tagPo.getId());
        tagDto.setLevel(tagPo.getLevel());
        tagDto.setName(tagPo.getName());
        tagDto.setPid(tagPo.getPid());
        tagDto.setRule(tagPo.getRule());
        return tagDto;
    }


    /**
     * 将TagDto转换为TagPo对象
     *
     * @return
     */
    private TagPo convert(TagDto tagDto) {
        TagPo tagPo = new TagPo();
        tagPo.setId(tagDto.getId());
        tagPo.setName(tagDto.getName());
        tagPo.setRule(tagDto.getRule());
        tagPo.setLevel(tagDto.getLevel());
        if (tagDto.getLevel() == 1) {
            //如果当前等级为1级,那么设置父ID为-1
            tagPo.setPid(-1L);
        } else {
            tagPo.setPid(tagDto.getPid());
        }
        tagPo.setCtime(new Date());
        tagPo.setUtime(new Date());
        return tagPo;
    }


}
